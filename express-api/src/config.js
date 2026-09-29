import path from 'node:path';
export function integer(name, fallback, min=1, max=1000000000) {
  const value=Number(process.env[name]??fallback);
  if(!Number.isInteger(value)||value<min||value>max)throw Error(`Invalid ${name}`);
  return value;
}
export const config={
  dir:path.resolve(process.env.DATA_DIR??'data'),
  maxJobs:integer('MAX_JOBS',200,1,10000),
  maxBytes:integer('MAX_UPLOAD_MB',20,1,50)*1024*1024,
  maxPixels:integer('MAX_PIXELS',12000000,1,20000000),
  threads:integer('CPU_THREADS',4,1,64),
  timeout:integer('JOB_TIMEOUT_SECONDS',180,5,3600)*1000,
  ttl:integer('RESULT_TTL_HOURS',24,1,168)*3600000,
};
