import test from 'node:test';
import assert from 'node:assert/strict';
import {explainFailure} from '../src/errors.js';
test('observed allocation failure is described as memory exhaustion',()=>{
  assert.match(explainFailure(new Error('OrtRun ERROR_CODE: 6 std::bad_alloc')),/memory/);
});
test('observed table failure is not misreported as a confirmed memory problem',()=>{
  assert.match(explainFailure('table index is out of bounds'),/compatibility/);
});
test('numeric runtime exception is not presented as a meaningful error code',()=>{
  assert.match(explainFailure(510135376),/failed internally/);
});
