"""Single-process, disk-backed FIFO. One worker owns one reusable model."""
import fcntl
import json
import logging
import sqlite3
import threading
import time
import uuid
from pathlib import Path


class JobFailure(RuntimeError):
    """A safe public message; the chained exception stays in server logs."""


class JobQueue:
    def __init__(self, directory, process, capacity=1000, retention=3600):
        if capacity < 1:
            raise ValueError("Queue capacity must be positive")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.owner = (self.directory / 'worker.lock').open('a')
        try:
            fcntl.flock(self.owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.owner.close()
            raise RuntimeError('Job directory already in use. Run exactly one Uvicorn worker.')
        self.path = self.directory / 'jobs.sqlite3'
        self.process = process
        self.capacity = capacity
        self.retention = retention
        self.stop = threading.Event()
        with self.connect() as db:
            db.execute('PRAGMA journal_mode=WAL')
            db.execute('CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, created REAL, '
                       'updated REAL, status TEXT, options TEXT, input BLOB, result BLOB, error TEXT)')
            # A killed process may have left an unfinished job; retry on restart.
            db.execute("UPDATE jobs SET status='queued' WHERE status='processing'")
        self.thread = threading.Thread(target=self.run, daemon=True, name='image-worker')
        self.thread.start()

    def connect(self):
        # closing is explicit: sqlite connection context managers only commit/rollback.
        from contextlib import contextmanager

        @contextmanager
        def connection():
            db = sqlite3.connect(self.path, timeout=30)
            try:
                with db:
                    yield db
            finally:
                db.close()
        return connection()

    def cleanup(self, db):
        db.execute("DELETE FROM jobs WHERE status IN ('completed','failed') AND updated < ?",
                   (time.time() - self.retention,))

    def submit(self, data, options):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            self.cleanup(db)
            count = db.execute("SELECT count(*) FROM jobs WHERE status IN ('queued','processing')").fetchone()[0]
            stored = db.execute('SELECT coalesce(sum(coalesce(length(input),0)+coalesce(length(result),0)),0) FROM jobs').fetchone()[0]
            if count >= self.capacity or stored + len(data or b'') > 2 * 1024**3:
                raise OverflowError('Queue storage or job limit reached. Try again later.')
            job_id = uuid.uuid4().hex
            now = time.time()
            db.execute('INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?)',
                       (job_id, now, now, 'queued', json.dumps(options), data, None, None))
        logging.getLogger("uvicorn.error").info("Queue accepted %s (%s/%s active or waiting)", job_id, count + 1, self.capacity)
        return job_id

    def get(self, job_id, result=False):
        with self.connect() as db:
            row = db.execute('SELECT status,error,updated FROM jobs WHERE id=?', (job_id,)).fetchone()
            if row is None or (row[0] in ('completed', 'failed') and row[2] < time.time()-self.retention):
                return None
            if result:
                return db.execute('SELECT result FROM jobs WHERE id=?', (job_id,)).fetchone()[0]
            return {'jobId': job_id, 'status': row[0], 'error': row[1],
                    'statusUrl': f'/jobs/{job_id}', 'resultUrl': f'/jobs/{job_id}/result'}

    def run(self):
        while not self.stop.is_set():
            try:
                with self.connect() as db:
                    self.cleanup(db)
                    row = db.execute("SELECT id,input,options FROM jobs WHERE status='queued' ORDER BY created,rowid LIMIT 1").fetchone()
                    if row:
                        db.execute("UPDATE jobs SET status='processing',updated=? WHERE id=?", (time.time(), row[0]))
                if not row:
                    self.stop.wait(.25)
                    continue
                job_id, data, options = row
                started = time.monotonic()
                logging.getLogger("uvicorn.error").info("Queue processing %s", job_id)
                try:
                    result = self.process(data, json.loads(options))
                    with self.connect() as db:
                        db.execute("UPDATE jobs SET status='completed',result=?,input=NULL,options='{}',updated=? WHERE id=?",
                                   (result, time.time(), job_id))
                    logging.getLogger("uvicorn.error").info("Queue completed %s in %.2fs", job_id, time.monotonic() - started)
                except Exception as exc:
                    logging.exception('Image job %s failed', job_id)
                    with self.connect() as db:
                        db.execute("UPDATE jobs SET status='failed',input=NULL,options='{}',error=?,updated=? WHERE id=?",
                                   (str(exc) if isinstance(exc, JobFailure) else 'Image processing failed; check server logs using this job ID.', time.time(), job_id))
            except Exception:
                logging.exception('Queue worker error')
                self.stop.wait(1)

    def close(self):
        self.stop.set()
        self.thread.join()
        self.owner.close()
