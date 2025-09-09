
const API = (window.API_BASE || "http://localhost:8000");
let token = localStorage.getItem("titapa_token") || "";

function setAuth(t) {
  token = t;
  localStorage.setItem("titapa_token", t);
  document.getElementById("authStatus").innerText = token ? "مصدق ✅ / Authenticated" : "غير مسجل";
  refreshKpis();
}

async function register(){
  const email = document.getElementById("email").value;
  const password = document.getElementById("password").value;
  const res = await fetch(API + "/auth/register", {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({email,password})});
  const data = await res.json();
  if(res.ok){ setAuth(data.token); } else { alert(data.detail||"خطأ") }
}
async function login(){
  const email = document.getElementById("email").value;
  const password = document.getElementById("password").value;
  const res = await fetch(API + "/auth/login", {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({email,password})});
  const data = await res.json();
  if(res.ok){ setAuth(data.token); } else { alert(data.detail||"خطأ") }
}
async function upload(){
  if(!token) return alert("سجّل الدخول أولًا");
  const f = document.getElementById("file").files[0];
  if(!f) return alert("اختر ملف CSV");
  const fd = new FormData(); fd.append("file", f);
  const res = await fetch(API + "/data/upload", {method:"POST", headers:{Authorization:"Bearer "+token}, body:fd});
  const data = await res.json();
  document.getElementById("uploadStatus").innerText = JSON.stringify(data);
  if(data.dataset_id) document.getElementById("datasetId").value = data.dataset_id;
}
async function train(){
  if(!token) return alert("سجّل الدخول أولًا");
  const dataset_id = parseInt(document.getElementById("datasetId").value||"0");
  const target = document.getElementById("target").value;
  const res = await fetch(API + "/ml/train", {method:"POST", headers:{"Content-Type":"application/json", Authorization:"Bearer "+token}, body: JSON.stringify({dataset_id, target})});
  const data = await res.json();
  document.getElementById("trainStatus").innerText = JSON.stringify(data);
  if(data.model_id) document.getElementById("modelId").value = data.model_id;
  refreshKpis();
}
async function predict(){
  if(!token) return alert("سجّل الدخول أولًا");
  const model_id = parseInt(document.getElementById("modelId").value||"0");
  let feat = {};
  try{ feat = JSON.parse(document.getElementById("features").value||"{}"); }catch(e){ return alert("صيغة JSON غير صحيحة") }
  const res = await fetch(API + "/ml/predict-json/" + model_id, {method:"POST", headers:{"Content-Type":"application/json", Authorization:"Bearer "+token}, body: JSON.stringify(feat)});
  const data = await res.json();
  document.getElementById("predictStatus").innerText = JSON.stringify(data);
}
async function refreshKpis(){
  if(!token) return;
  const res = await fetch(API + "/dashboards/kpis", {headers:{Authorization: "Bearer "+token}});
  const data = await res.json();
  document.getElementById("kpiUsers").innerText = data.users ?? "-";
  document.getElementById("kpiDatasets").innerText = data.datasets ?? "-";
  document.getElementById("kpiModels").innerText = data.models ?? "-";
}
async function listDatasets(){
  if(!token) return alert("سجّل الدخول أولًا");
  const res = await fetch(API + "/datasets", {headers:{Authorization:"Bearer "+token}});
  const data = await res.json();
  document.getElementById("datasetsBox").innerText = JSON.stringify(data, null, 2);
}

document.querySelectorAll(".control-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    const items = document.querySelectorAll(".visualization-item");
    if (btn.dataset.action === "pause") items.forEach(i => i.style.animationPlayState = "paused");
    if (btn.dataset.action === "play") items.forEach(i => i.style.animationPlayState = "running");
  });
});

if(token){ document.getElementById("authStatus").innerText = "مصدق ✅ / Authenticated"; refreshKpis(); }
