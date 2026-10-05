const runBtn = document.getElementById("runBtn");
const downloadBtn = document.getElementById("downloadBtn");
const statusEl = document.getElementById("status");
const rowsEl = document.getElementById("rows");
const parityEl = document.getElementById("parity");

const FP32_URL = "./model/v2/best_model_fp32.onnx";
const INT8_URL = "./model/v2/best_model_int8.onnx";
const META_URL = "./model/v2/metadata.json";
const N = 80;
const WARMUP = 10;
let latestReport = null;

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/";

function seededRandom(seed = 20261005) {
  let s = seed >>> 0;
  return () => {
    s = (1664525 * s + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

function makeInput(shape) {
  const rand = seededRandom();
  const arr = new Float32Array(shape.reduce((a,b)=>a*b,1));
  const featureDim = shape[2];
  const valid = Math.min(72, shape[1]);
  for (let i=0; i<valid*featureDim; i++) arr[i]=(rand()-0.5)*0.36;
  return arr;
}

function stats(xs) {
  const a=[...xs].sort((x,y)=>x-y);
  const mean=a.reduce((s,x)=>s+x,0)/a.length;
  const m=Math.floor(a.length/2);
  const median=a.length%2?a[m]:(a[m-1]+a[m])/2;
  const p95=a[Math.min(a.length-1,Math.ceil(a.length*0.95)-1)];
  return {mean,median,p95};
}

function bytesText(n) {
  if (n<1024) return n+" B";
  if (n<1024*1024) return (n/1024).toFixed(1)+" KB";
  return (n/(1024*1024)).toFixed(2)+" MB";
}

async function fetchJson(url) {
  const r=await fetch(url,{cache:"no-cache"});
  if(!r.ok) throw new Error(url+": HTTP "+r.status);
  return r.json();
}

async function fetchBytes(url) {
  const r=await fetch(url,{cache:"no-cache"});
  if(!r.ok) throw new Error(url+": HTTP "+r.status);
  const buf=await r.arrayBuffer();
  return {buf,bytes:buf.byteLength};
}

async function bench(session,input,shape) {
  const name=session.inputNames[0];
  const tensor=new ort.Tensor("float32",input,shape);
  for(let i=0;i<WARMUP;i++) await session.run({[name]:tensor});
  const times=[];
  let output=null;
  for(let i=0;i<N;i++){
    const t0=performance.now();
    const result=await session.run({[name]:tensor});
    times.push(performance.now()-t0);
    output=Float32Array.from(result[session.outputNames[0]].data);
  }
  return {times,output};
}

function diff(a,b) {
  let max=0,sum=0;
  const n=Math.min(a.length,b.length);
  for(let i=0;i<n;i++){
    const d=Math.abs(a[i]-b[i]);
    sum+=d;if(d>max)max=d;
  }
  return {mean:sum/n,max};
}

function argmax(a) {
  let bi=0;
  for(let i=1;i<a.length;i++) if(a[i]>a[bi]) bi=i;
  return bi;
}

function deviceInfo() {
  return {
    platform:navigator.userAgentData?.platform||navigator.platform||"unknown",
    mobile:navigator.userAgentData?.mobile??/Android|iPhone|iPad|Mobile/i.test(navigator.userAgent),
    hardwareConcurrency:navigator.hardwareConcurrency||null,
    deviceMemoryGB:navigator.deviceMemory||null,
    webgpuAvailable:Boolean(navigator.gpu),
    viewport:{width:innerWidth,height:innerHeight,dpr:devicePixelRatio||1},
  };
}

function download(payload) {
  const blob=new Blob([JSON.stringify(payload,null,2)],{type:"application/json"});
  const url=URL.createObjectURL(blob);
  const a=document.createElement("a");
  a.href=url;a.download=`ieum_v2_benchmark_${Date.now()}.json`;a.click();
  setTimeout(()=>URL.revokeObjectURL(url),1000);
}

downloadBtn.addEventListener("click",()=>{if(latestReport)download(latestReport);});

runBtn.addEventListener("click",async()=>{
  runBtn.disabled=true;downloadBtn.disabled=true;latestReport=null;
  rowsEl.innerHTML="";parityEl.textContent="-";
  try{
    statusEl.textContent=" v2 모델 확인 중...";
    const meta=await fetchJson(META_URL);
    if(meta.feature_schema!=="ieum_v2_190") throw new Error("ieum_v2_190 metadata가 아닙니다.");
    const shape=[1,Number(meta.max_sequence_length||100),Number(meta.feature_dim||190)];
    const fetchStart=performance.now();
    const [fpFile,iqFile]=await Promise.all([fetchBytes(FP32_URL),fetchBytes(INT8_URL)]);
    const fetchMs=performance.now()-fetchStart;

    let t=performance.now();
    const fp=await ort.InferenceSession.create(fpFile.buf.slice(0),{executionProviders:["wasm"],graphOptimizationLevel:"all"});
    const fpLoad=performance.now()-t;
    t=performance.now();
    const iq=await ort.InferenceSession.create(iqFile.buf.slice(0),{executionProviders:["wasm"],graphOptimizationLevel:"all"});
    const iqLoad=performance.now()-t;

    let wg=null,wgLoad=null,wgError=null;
    if(navigator.gpu){
      try{
        t=performance.now();
        wg=await ort.InferenceSession.create(fpFile.buf.slice(0),{executionProviders:["webgpu"],graphOptimizationLevel:"all"});
        wgLoad=performance.now()-t;
      }catch(e){wgError=String(e?.message||e);}
    }

    const input=makeInput(shape);
    statusEl.textContent=" WASM FP32 측정 중...";
    const fpRes=await bench(fp,input,shape);
    statusEl.textContent=" WASM INT8 측정 중...";
    const iqRes=await bench(iq,input,shape);
    let wgRes=null;
    if(wg){
      statusEl.textContent=" WebGPU FP32 측정 중...";
      try{wgRes=await bench(wg,input,shape);}catch(e){wgError=String(e?.message||e);}
    }

    const rows=[
      ["v2 ONNX WASM FP32",fetchMs+fpLoad,stats(fpRes.times),fpFile.bytes],
      ["v2 ONNX WASM INT8",fetchMs+iqLoad,stats(iqRes.times),iqFile.bytes],
    ];
    if(wgRes) rows.push(["v2 ONNX WebGPU FP32",fetchMs+wgLoad,stats(wgRes.times),fpFile.bytes]);

    rowsEl.innerHTML=rows.map(([name,load,s,bytes])=>
      `<tr><td>${name}</td><td>${load.toFixed(1)}</td><td>${s.mean.toFixed(2)}</td><td>${s.median.toFixed(2)}</td><td>${s.p95.toFixed(2)}</td><td>${bytesText(bytes)}</td></tr>`
    ).join("");

    const delta=diff(fpRes.output,iqRes.output);
    const parity={
      fp32_vs_int8:{
        mean_abs_error:delta.mean,
        max_abs_error:delta.max,
        argmax_match:argmax(fpRes.output)===argmax(iqRes.output),
      },
      webgpu:{available:Boolean(navigator.gpu),session_created:Boolean(wg),error:wgError},
      note:"합성 입력 runtime 비교입니다. 실제 정확도 판단에는 held-out test ONNX 평가를 사용합니다."
    };
    parityEl.textContent=JSON.stringify(parity,null,2);
    latestReport={
      format:"ieum-browser-benchmark-v2",
      createdAt:new Date().toISOString(),
      metadata:{feature_schema:meta.feature_schema,selected_model:meta.selected_model,labels:meta.labels},
      inputShape:shape,iterations:N,warmup:WARMUP,device:deviceInfo(),
      engines:rows.map(([name,load,s,bytes])=>({
        name,loadMs:load,latencyMeanMs:s.mean,latencyMedianMs:s.median,latencyP95Ms:s.p95,modelBytes:bytes
      })),
      parity,
    };
    downloadBtn.disabled=false;statusEl.textContent=" 완료";
  }catch(e){
    console.error(e);
    statusEl.textContent=" 실행 불가";
    parityEl.textContent=String(e?.stack||e)+"\n\nv2 학습/ONNX 내보내기가 아직 완료되지 않았다면 정상입니다.";
  }finally{runBtn.disabled=false;}
});
