export function explainFailure(error) {
  const detail=String(error?.message??error);
  if(/bad_alloc|out of memory|memory access out of bounds|cannot enlarge memory/i.test(detail))
    return 'The model ran out of browser runtime memory. Close other heavy tabs and release model memory before trying again. See the technical log for the failing stage.';
  if(/table index is out of bounds|too many storage buffers/i.test(detail))
    return 'The model hit a browser inference compatibility failure. If using GPU, try Local CPU. See the technical log for details.';
  if(/^\d+$/.test(detail))
    return 'The inference runtime failed internally. Its numeric error is not a processing result. See the technical log for details.';
  return detail;
}
