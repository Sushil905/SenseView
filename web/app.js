const themeToggle=document.querySelector('#theme-toggle');
const themeMeta=document.querySelector('meta[name="theme-color"]');
const savedTheme=localStorage.getItem('senseview.theme');
function setTheme(theme){
  const daylight=theme==='daylight';
  document.documentElement.dataset.theme=daylight?'daylight':'galaxy';
  themeToggle.setAttribute('aria-label',daylight?'Switch to galaxy theme':'Switch to daylight theme');
  themeToggle.setAttribute('aria-pressed',String(daylight));
  themeToggle.querySelector('.theme-icon').textContent=daylight?'☾':'☼';
  themeToggle.querySelector('.theme-name').textContent=daylight?'Galaxy':'Daylight';
  themeMeta.content=daylight?'#f2f2ff':'#080b1d';
}
setTheme(savedTheme==='daylight'?'daylight':'galaxy');
themeToggle.addEventListener('click',()=>{
  const theme=document.documentElement.dataset.theme==='daylight'?'galaxy':'daylight';
  setTheme(theme);localStorage.setItem('senseview.theme',theme);
});

const health=document.querySelector('#health');
const form=document.querySelector('#predict-form');
const output=document.querySelector('#output');
const runButton=document.querySelector('#run-btn');
const sampleButton=document.querySelector('#sample-btn');
const inputs=Array.from(form.querySelectorAll('input[type=file]'));
const HISTORY_KEY='senseview.recentResults.v1';

function goToSection(id,updateHash=true){
  const section=document.getElementById(id);
  if(!section)return;
  document.querySelectorAll('.app-section').forEach(node=>node.classList.toggle('active',node===section));
  document.querySelectorAll('.nav-tab').forEach(button=>{
    const active=button.dataset.section===id;
    button.classList.toggle('active',active);
    button.setAttribute('aria-current',active?'page':'false');
  });
  if(updateHash)history.replaceState(null,'','#'+id);
  window.scrollTo({top:0,behavior:'smooth'});
}
document.querySelectorAll('.nav-tab').forEach(button=>button.addEventListener('click',()=>goToSection(button.dataset.section)));
document.querySelectorAll('[data-go]').forEach(button=>button.addEventListener('click',()=>goToSection(button.dataset.go)));
window.addEventListener('hashchange',()=>goToSection(location.hash.slice(1),false));
if(location.hash&&document.getElementById(location.hash.slice(1)))goToSection(location.hash.slice(1),false);

fetch('/api/health').then(r=>{if(!r.ok)throw Error();return r.json()})
  .then(()=>{health.classList.add('live');health.innerHTML='<i></i>FastAPI online'})
  .catch(()=>{health.innerHTML='<i></i>Service unavailable'});

fetch('/api/status').then(r=>r.json()).then(status=>{
  const raf=document.querySelector('#rafdb-status'),rafDetail=document.querySelector('#rafdb-detail');
  raf.textContent=status.rafdb_ready?'RAF-DB files detected':'RAF-DB not configured';
  raf.classList.toggle('online',status.rafdb_ready);
  rafDetail.textContent=status.rafdb_ready?'Aligned images and annotations found.':'Add the licensed RAF-DB files to data/raw/rafdb/.';
  document.querySelector('#retina-status').textContent=status.retinaface_available?'Installed and ready':'Install requirements-vision.txt to enable';
  document.querySelector('#svm-status').textContent=status.svm_checkpoint_ready?'Session SVM checkpoint detected':'No SVM checkpoint loaded';
  document.querySelector('#temporal-status').textContent=status.temporal_checkpoint_ready?'ViT/LSTM checkpoint detected':'Train/export a temporal checkpoint first';
  document.querySelector('#clip-status').textContent=status.clip_manifest_ready?'Processed clip manifest detected':'Preprocess clips to add face crops and yaw';
}).catch(()=>{});

function updateInputs(){
  const counts=inputs.map(input=>input.files.length),selected=counts.every(count=>count>0);
  const synchronized=selected&&counts.every(count=>count===counts[0]);
  runButton.disabled=!synchronized;
  runButton.textContent=selected&&!synchronized?'Frame counts must match':'Analyze synchronized clip  →';
  inputs.forEach(input=>{
    const tile=input.closest('.upload'),files=Array.from(input.files),preview=tile.querySelector('img');
    tile.classList.toggle('has-file',files.length>0);
    tile.querySelector('.file-state').textContent=files.length
      ? files.length+' frame'+(files.length===1?'':'s')+' selected':'Choose synchronized frames';
    if(preview.dataset.url)URL.revokeObjectURL(preview.dataset.url);
    if(files.length){preview.dataset.url=URL.createObjectURL(files[0]);preview.src=preview.dataset.url}
    else preview.removeAttribute('src');
  });
}
inputs.forEach(input=>input.addEventListener('change',updateInputs));

function makeBar(label,value,extra){
  const row=document.createElement('div');row.className=('quality-row '+(extra||'')).trim();
  const name=document.createElement('span');name.textContent=label;
  const bar=document.createElement('div');bar.className='bar';
  const fill=document.createElement('i');fill.style.width=(Math.max(0,Math.min(1,value))*100)+'%';bar.append(fill);
  const amount=document.createElement('b');amount.textContent=(value*100).toFixed(0)+'%';
  row.append(name,bar,amount);return row;
}
function heading(leftText,rightText){
  const node=document.createElement('div');node.className='section-title';
  const left=document.createElement('span'),right=document.createElement('span');
  left.textContent=leftText;right.textContent=rightText;node.append(left,right);return node;
}
function renderResult(data){
  const labels=data.expressions||[],scores=data.probabilities||[],weights=data.view_quality||[];
  const yaw=data.fusion_method==='lowest_yaw';output.replaceChildren();
  const top=document.createElement('div');top.className='result-top';
  const ring=document.createElement('div');ring.className='confidence-ring';
  ring.style.setProperty('--progress',(Math.max(0,Math.min(1,data.confidence))*100)+'%');
  const pct=document.createElement('span');pct.textContent=(data.confidence*100).toFixed(0)+'%';ring.append(pct);
  const copy=document.createElement('div');copy.className='result-copy';
  const small=document.createElement('small');small.textContent='Top model estimate';
  const title=document.createElement('h4');title.textContent=data.expression;
  const detail=document.createElement('p');detail.textContent=yaw
    ? 'Selected '+((data.selected_views||[]).join(' → ')||'lowest-yaw views')
    : 'Synthetic sample · learned camera weights';
  if(data.session_classifier)detail.textContent+=' · SVM '+data.session_classifier.prediction;
  copy.append(small,title,detail);top.append(ring,copy);output.append(top);
  output.append(heading(yaw?'Selected camera per view':'Learned camera weights',yaw?'LOWEST |YAW|':'FUSION WEIGHTS'));
  weights.forEach((value,index)=>output.append(makeBar('Camera '+String(index+1).padStart(2,'0'),value)));
  output.append(heading('Expression score distribution','MODEL SCORES'));
  const list=document.createElement('div');list.className='dist-list';
  labels.map((label,index)=>({label:label,value:scores[index]||0})).sort((a,b)=>b.value-a.value)
    .forEach((item,index)=>list.append(makeBar(item.label,item.value,'dist-row'+(index===0?' top':''))));
  output.append(list);const foot=document.createElement('p');foot.className='result-notice';
  foot.textContent=data.notice||'Research output only.';output.append(foot);
}

function readHistory(){try{return JSON.parse(localStorage.getItem(HISTORY_KEY)||'[]')}catch{return []}}
function saveHistory(data,source){
  const entries=readHistory();
  entries.unshift({created_at:new Date().toISOString(),source:source,expression:data.expression,
    confidence:data.confidence,result:data});
  try{localStorage.setItem(HISTORY_KEY,JSON.stringify(entries.slice(0,20)))}catch{}
  renderHistory();
  document.querySelector('#result-time').textContent=new Date().toLocaleString();
}
function renderHistory(){
  const host=document.querySelector('#history-list'),entries=readHistory();host.replaceChildren();
  if(!entries.length){
    const empty=document.createElement('div');empty.className='empty-state history-empty';
    empty.innerHTML='<div class="empty-glyph">◷</div><strong>No analyses yet</strong><p>Completed result summaries will appear here.</p>';
    host.append(empty);return;
  }
  entries.forEach(entry=>{
    const row=document.createElement('button');row.type='button';row.className='history-entry';
    const expression=document.createElement('span');expression.className='history-expression';expression.textContent=entry.expression;
    const meta=document.createElement('span');meta.className='history-meta';
    meta.textContent=new Date(entry.created_at).toLocaleString()+' · '+entry.source;
    const confidence=document.createElement('span');confidence.className='history-confidence';
    confidence.textContent=(entry.confidence*100).toFixed(1)+'%';
    row.append(expression,meta,confidence);
    row.addEventListener('click',()=>{renderResult(entry.result);document.querySelector('#result-time').textContent=new Date(entry.created_at).toLocaleString();goToSection('results')});
    host.append(row);
  });
}
document.querySelector('#clear-history').addEventListener('click',()=>{
  localStorage.removeItem(HISTORY_KEY);renderHistory();
});
renderHistory();

async function request(url,options){
  runButton.disabled=true;sampleButton.disabled=true;
  output.innerHTML='<div class="empty-state"><div class="empty-glyph">…</div><strong>Analyzing synchronized views…</strong><p>Running locally</p></div>';
  try{
    const response=await fetch(url,options),data=await response.json();
    if(!response.ok)throw Error(data.detail||data.error||'Request failed.');
    renderResult(data);saveHistory(data,url.includes('/demo')?'Synthetic sample':'Uploaded clip');
    goToSection('results');
  }catch(error){
    output.replaceChildren();const message=document.createElement('p');
    message.className='result-notice';message.textContent=error.message||'Local inference failed.';output.append(message);
    goToSection('results');
  }finally{updateInputs();sampleButton.disabled=false}
}
form.addEventListener('submit',event=>{
  event.preventDefault();const counts=inputs.map(input=>input.files.length);
  if(!counts[0]||!counts.every(count=>count===counts[0]))return;
  request('/api/predict',{method:'POST',body:new FormData(form)});
});
sampleButton.addEventListener('click',()=>request('/api/demo'));
document.querySelectorAll('[data-sample]').forEach(button=>button.addEventListener('click',()=>request('/api/demo')));
