// Emscripten's JS binding glue uses Function constructors. MV3 disallows them.
// Replace just the two glue factories with ordinary closures; WASM is unchanged.
export function makeCspCompatible(source) {
  function replaceBetween(start, end, replacement) {
    const a=source.indexOf(start), b=source.indexOf(end,a);
    if(a<0||b<0) throw Error('OpenCV binding layout changed; review CSP patch.');
    source=source.slice(0,a)+replacement+source.slice(b);
  }
  replaceBetween('function createJsInvoker(', 'function craftInvokerFunction(', `
function createJsInvoker(argTypes,isClassMethodFunc,returns,isAsync){
  var needsStack=usesDestructorStack(argTypes), count=argTypes.length-2;
  return function(humanName,throwBindingError,invoker,fn,runDestructors,fromRetWire,toClassParamWire,...bindings){
    return function(...args){
      var destructors=needsStack?[]:null, wired=[fn];
      if(isClassMethodFunc) wired.push(toClassParamWire(destructors,this));
      for(var i=0;i<count;i++) wired.push(bindings[i](destructors,args[i]));
      var rv=invoker(...wired);
      if(needsStack) runDestructors(destructors);
      else {
        var d=count;
        for(var i=isClassMethodFunc?1:2;i<argTypes.length;i++){
          if(argTypes[i].destructorFunction!==null) bindings[d++](wired[i-(isClassMethodFunc?0:1)]);
        }
      }
      if(returns) return fromRetWire(rv);
    };
  };
}
`);
  replaceBetween('var __emval_create_invoker=', 'var __emval_get_property=', `
var __emval_create_invoker=(argCount,argTypesPtr,kind)=>{
  var [retType,...argTypes]=emval_lookupTypes(argCount,argTypesPtr);
  var toReturnWire=retType.toWireType.bind(retType);
  var readers=argTypes.map(type=>type.readValueFromPointer.bind(type));
  var invoker=function(handle,methodName,destructorsRef,args){
    var values=readers.map((read,i)=>read(args+i*8)), result;
    if(kind===0) result=Emval.toValue(handle)(...values);
    else if(kind===1) result=Emval.toValue(handle)[getStringOrSymbol(methodName)](...values);
    else if(kind===2) result=Reflect.construct(Emval.toValue(handle),values);
    else if(kind===3) result=values[values.length-1];
    else throw Error('Unsupported emval invocation kind');
    if(!retType.isVoid) return emval_returnValue(toReturnWire,destructorsRef,result);
  };
  return emval_addMethodCaller(invoker);
};
`);
  if(source.includes('new Function(')) throw Error('Unexpected dynamic code in OpenCV runtime.');
  return source;
}
