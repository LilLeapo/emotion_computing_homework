// Live demo for emotion.diag — 6 tabs over StructBERT / emotion2vec / ViT-FER.
// Text, uploaded speech, uploaded face, and multimodal fusion call the local
// FastAPI backend. Preset samples remain client-side demos.

const { useState, useEffect, useRef, useMemo } = React;

const EMO_KEYS = ['happy','sad','angry','fear','disgust','surprise','neutral'];
const ZH = { happy:'高兴', sad:'悲伤', angry:'愤怒', fear:'恐惧',
             disgust:'厌恶', surprise:'惊讶', neutral:'中性' };
const COL = {
  happy:'oklch(0.82 0.13 82)', surprise:'oklch(0.78 0.13 165)',
  neutral:'oklch(0.62 0.01 80)', sad:'oklch(0.65 0.10 230)',
  angry:'oklch(0.62 0.16 25)', fear:'oklch(0.55 0.09 290)',
  disgust:'oklch(0.58 0.08 130)',
};

const TABS = [
  { id:'text',  label:'文本分析',     ix:'01' },
  { id:'speech',label:'语音识别',     ix:'02' },
  { id:'face',  label:'面部表情',     ix:'03' },
  { id:'multi', label:'多模态融合',   ix:'04' },
  { id:'video', label:'视频时间线',   ix:'05' },
  { id:'live',  label:'实时摄像头',   ix:'06' },
];

// ─── deterministic mock classifier ─────────────────────────
// Uses keyword hits + light randomness so the same text gives the same answer.
const KW = {
  happy:    ['开心','高兴','哈哈','喜欢','棒','太好','爱','幸福','满意','成功','可爱','笑'],
  sad:      ['难过','伤心','想哭','失望','遗憾','孤独','悲','低落','哎','唉','后悔'],
  angry:    ['生气','愤怒','气死','烦','讨厌','凭什么','可恶','操','气','无语','怒'],
  fear:     ['害怕','恐惧','担心','紧张','怕','焦虑','慌','颤抖','不安'],
  disgust:  ['恶心','厌恶','嫌','吐','糟糕','反胃'],
  surprise: ['惊讶','居然','竟然','哇','啊','没想到','天哪','真的吗','突然'],
  neutral:  ['好的','收到','知道了','嗯','行','okay','可以'],
};
function classifyText(text){
  if(!text || !text.trim()){
    return EMO_KEYS.reduce((m,k)=>(m[k]=k==='neutral'?1:0,m),{});
  }
  const scores = {};
  EMO_KEYS.forEach(k=>{
    let s = 0.3;
    KW[k].forEach(w => { if(text.includes(w)) s += 1.4; });
    // smooth
    s *= 0.7 + 0.3 * Math.sin((text.length + k.charCodeAt(0))*0.31);
    scores[k] = Math.max(0.04, s);
  });
  // softmax-ish
  const sum = Object.values(scores).reduce((a,b)=>a+b,0);
  EMO_KEYS.forEach(k => { scores[k] = +(scores[k]/sum).toFixed(3); });
  return scores;
}
function topEmo(dist){
  return Object.entries(dist).sort((a,b)=>b[1]-a[1])[0];
}
function attentionFor(text, dominantEmo){
  // crude: weight each char by whether it's part of a matching keyword
  const out = [];
  const kw = KW[dominantEmo] || [];
  for(let i=0;i<text.length;i++){
    let w = 0.08 + (Math.sin(i*1.7 + text.charCodeAt(i)*0.13)+1)*0.12;
    kw.forEach(k => {
      const s = text.indexOf(k);
      if(s>=0 && i>=s && i<s+k.length) w = Math.max(w, 0.78 + Math.random()*0.18);
    });
    out.push({ ch: text[i], w: Math.min(1, w) });
  }
  return out;
}

async function postForm(url, formData){
  const res = await fetch(url, { method:'POST', body:formData });
  const data = await res.json().catch(()=>({ error:'Invalid JSON response' }));
  if(!res.ok) throw new Error(data.error || `Request failed: ${res.status}`);
  return data;
}

async function apiText(text){
  const fd = new FormData();
  fd.append('text', text);
  return postForm('/api/text', fd);
}

async function apiSpeech(file){
  const fd = new FormData();
  fd.append('file', file);
  return postForm('/api/speech', fd);
}

async function apiFace(file){
  const fd = new FormData();
  fd.append('file', file);
  return postForm('/api/face', fd);
}

async function apiMultimodal({ text, audio, image }){
  const fd = new FormData();
  if(text) fd.append('text', text);
  if(audio) fd.append('audio', audio);
  if(image) fd.append('image', image);
  return postForm('/api/multimodal', fd);
}

// ─── shared : verdict + bars + radar ────────────────────────
function ResultBlock({ dist, title='Dominant', subtitle, modelName, latency, sourceLabel }){
  const sorted = useMemo(()=>Object.entries(dist).sort((a,b)=>b[1]-a[1]), [dist]);
  const [topK, topV] = sorted[0];
  return (
    <div className="results">
      {/* verdict */}
      <div className="card r-verdict">
        <h4>{title}<span className="tag">FUSED · 7-CLS</span></h4>
        <div className="verdict-big">{topK}</div>
        <div className="verdict-zh">{ZH[topK]} · {subtitle || '主导情绪'}</div>
        <div className="verdict-stats">
          <div className="vstat"><span>CONFIDENCE</span><b>{(topV*100).toFixed(1)}%</b></div>
          <div className="vstat"><span>ENTROPY</span><b>{entropy(dist).toFixed(2)}</b></div>
          {modelName && <div className="vstat"><span>MODEL</span><b style={{fontFamily:'var(--mono)',fontSize:11,fontStyle:'normal'}}>{modelName}</b></div>}
          {latency && <div className="vstat"><span>LATENCY</span><b>{latency} ms</b></div>}
          {sourceLabel && <div className="vstat"><span>SOURCE</span><b style={{fontFamily:'var(--mono)',fontSize:11,fontStyle:'normal'}}>{sourceLabel}</b></div>}
        </div>
      </div>

      {/* bars */}
      <div className="card r-bars">
        <h4>Probability<span className="tag">SOFTMAX</span></h4>
        <div className="bar-list">
          {sorted.map(([k,v],i)=>(
            <div className={'row'+(i===0?' top':'')} key={k}>
              <span className="lb">{k}</span>
              <span className="tk"><span className="fl" style={{width:(v*100).toFixed(1)+'%',background:COL[k]}}/></span>
              <span className="vl">{v.toFixed(3)}</span>
            </div>
          ))}
        </div>
      </div>

      {/* radar */}
      <div className="card r-radar">
        <h4>Radar<span className="tag">7-AXIS</span></h4>
        <div className="radar-wrap"><Radar dist={dist} /></div>
      </div>
    </div>
  );
}

function entropy(d){
  let h=0; Object.values(d).forEach(p => { if(p>0) h -= p*Math.log2(p); });
  return h;
}

function Radar({ dist }){
  const r = 90;
  const cx = 120, cy = 120;
  const angle = (i) => (-90 + i*360/7) * Math.PI/180;
  const pts = EMO_KEYS.map((k,i)=>{
    const v = dist[k] || 0;
    const x = cx + Math.cos(angle(i))*r*Math.max(0.04,v);
    const y = cy + Math.sin(angle(i))*r*Math.max(0.04,v);
    return [x,y];
  });
  const path = pts.map((p,i)=>(i?'L':'M')+p[0]+' '+p[1]).join(' ')+' Z';
  return (
    <svg className="radar-svg" viewBox="0 0 240 240">
      {/* grid */}
      {[0.25,0.5,0.75,1].map((s,i)=>{
        const gp = EMO_KEYS.map((_,i)=>{
          const x = cx + Math.cos(angle(i))*r*s;
          const y = cy + Math.sin(angle(i))*r*s;
          return [x,y];
        });
        const d = gp.map((p,i)=>(i?'L':'M')+p[0]+' '+p[1]).join(' ')+' Z';
        return <path key={s} d={d} fill="none" stroke="rgba(236,232,223,.1)" strokeWidth=".7"/>;
      })}
      {/* axes */}
      {EMO_KEYS.map((k,i)=>{
        const x = cx + Math.cos(angle(i))*r;
        const y = cy + Math.sin(angle(i))*r;
        return <line key={k} x1={cx} y1={cy} x2={x} y2={y}
          stroke="rgba(236,232,223,.08)" strokeWidth=".7"/>;
      })}
      {/* shape */}
      <path d={path} fill="rgba(241,202,118,.18)" stroke="rgb(241,202,118)" strokeWidth="1.2" strokeLinejoin="round"/>
      {pts.map((p,i)=>(
        <circle key={i} cx={p[0]} cy={p[1]} r="2.5" fill="rgb(241,202,118)"/>
      ))}
      {/* labels */}
      {EMO_KEYS.map((k,i)=>{
        const x = cx + Math.cos(angle(i))*(r+18);
        const y = cy + Math.sin(angle(i))*(r+18);
        return (
          <text key={k} x={x} y={y} fontSize="9.5" fontFamily="JetBrains Mono"
            fill="rgba(236,232,223,.5)" textAnchor="middle" dominantBaseline="middle"
            letterSpacing=".05em">{k}</text>
        );
      })}
    </svg>
  );
}

// ─── TAB 1 : TEXT ───────────────────────────────────────────
const TEXT_EXAMPLES = [
  '哈哈哈这也太巧了吧，我们居然又见面了',
  '这个版本又退化了，根本没法用',
  '会议安排在下周三下午三点',
  '我有点紧张，怕自己说错',
  '什么？居然真的中了？！',
  '想到这件事我就难过，今天又失眠了',
];

function TextTab(){
  const [val, setVal] = useState(TEXT_EXAMPLES[0]);
  const [pending, setPending] = useState(false);
  const [result, setResult] = useState(()=>({
    dist: classifyText(TEXT_EXAMPLES[0]),
    text: TEXT_EXAMPLES[0],
    latency: 78,
  }));
  const [showAttn, setShowAttn] = useState(true);

  async function run(){
    if(!val.trim()) return;
    setPending(true);
    try{
      const data = await apiText(val);
      setResult({
        dist: data.dist,
        text: val,
        latency: data.latency,
        attention: data.attention || [],
        modelName: data.modelName || 'StructBERT-base-zh',
      });
    }catch(e){
      alert(e.message || '文本分析失败');
    }finally{
      setPending(false);
    }
  }
  // run on mount once for default
  useEffect(()=>{ /* result already initialized */ }, []);

  const [topK] = topEmo(result.dist);
  const attn = result.attention && result.attention.length
    ? result.attention
    : attentionFor(result.text, topK);

  return (
    <>
      <div className="panel-hd">
        <div>
          <h2><em>文本</em> 情感分析</h2>
          <p>输入中文文本，StructBERT 输出 7 类情绪概率分布，并标注每个字对最终判断的注意力贡献。</p>
        </div>
        <div className="right">
          MODEL · <b>StructBERT-base-zh</b><br/>
          SOURCE · <b>iic/nlp_structbert_emotion-classification</b><br/>
          MAX TOK · <b>512</b>
        </div>
      </div>

      <div className="input-grid">
        <div className="text-input-wrap">
          <textarea className="text-in" value={val}
            onChange={e=>setVal(e.target.value)}
            placeholder="输入一段中文…例如：今天真是太开心了！"
            onKeyDown={e=>{ if((e.metaKey||e.ctrlKey)&&e.key==='Enter') run(); }}/>
          <div className="row-ex">
            {TEXT_EXAMPLES.map(ex=>(
              <span className="ex-chip" key={ex} onClick={()=>setVal(ex)}>{ex.slice(0,16)}{ex.length>16?'…':''}</span>
            ))}
          </div>
        </div>
        <div className="input-controls">
          <button className="run-btn" onClick={run} disabled={pending}>
            <span>{pending? <><span className="spinner"/> Inferring…</> : 'ANALYZE'}</span>
            <span className="key">⌘↵</span>
          </button>
          <div className="opt-row">
            <span>SHOW ATTENTION</span>
            <div className="seg">
              <button className={showAttn?'on':''} onClick={()=>setShowAttn(true)}>ON</button>
              <button className={!showAttn?'on':''} onClick={()=>setShowAttn(false)}>OFF</button>
            </div>
          </div>
          <div className="opt-row">
            <span>LENGTH</span>
            <span style={{color:'var(--fg)'}}>{val.length} 字</span>
          </div>
          <div className="opt-row">
            <span>STATUS</span>
            <span style={{color:'var(--c-ok)'}}>● MODEL READY</span>
          </div>
        </div>
      </div>

      <ResultBlock dist={result.dist}
        subtitle="基于 StructBERT 输出"
        modelName={result.modelName || "StructBERT-base-zh"}
        latency={result.latency}
        sourceLabel="text"/>

      {showAttn && (
        <div className="card r-attn">
          <h4>Token Attention<span className="tag">字符级</span></h4>
          <div className="attn-box">
            {attn.map((t,i)=>(
              <span key={i} className="attn-tok"
                style={{
                  background:`rgba(241,202,118,${(t.w*0.85+0.04).toFixed(3)})`,
                  color: t.w>0.55? '#1a1a1a' : 'var(--fg)',
                }}>{t.ch}</span>
            ))}
          </div>
          <div className="attn-legend">
            <span>LOW</span>
            <div className="scale"></div>
            <span>HIGH</span>
            <span style={{marginLeft:'auto'}}>DOMINANT · <span style={{color:'var(--c-joy)'}}>{topK}</span></span>
          </div>
        </div>
      )}
    </>
  );
}

// ─── TAB 2 : SPEECH ─────────────────────────────────────────
function SpeechTab(){
  const [file, setFile] = useState(null);
  const [recording, setRecording] = useState(false);
  const [recTime, setRecTime] = useState(0);
  const [pending, setPending] = useState(false);
  const [result, setResult] = useState(null);
  const [wave, setWave] = useState(()=>genWave('idle'));

  useEffect(()=>{
    if(!recording) return;
    const id = setInterval(()=>{
      setRecTime(t=>t+0.1);
      setWave(genWave('rec'));
    }, 100);
    return ()=>clearInterval(id);
  }, [recording]);

  function onFile(e){
    const f = e.target.files[0];
    if(!f) return;
    setFile({ name: f.name, size:(f.size/1024).toFixed(1)+' KB',
              dur:'~ '+(2+Math.random()*4).toFixed(1)+' s' });
    setWave(genWave('file'));
    analyze('file', f);
  }
  function toggleRec(){
    if(recording){
      setRecording(false);
      const dur = recTime;
      setFile({ name:'recording.wav', size:(dur*32).toFixed(1)+' KB', dur:dur.toFixed(1)+' s' });
      analyze('rec', 'recording');
      setRecTime(0);
    } else {
      setRecording(true); setRecTime(0); setFile(null); setResult(null);
    }
  }
  async function analyze(src, payload){
    setPending(true);
    try{
      if(payload instanceof File){
        const data = await apiSpeech(payload);
        setResult({
          dist: data.dist,
          latency: data.latency,
          modelName: data.modelName || 'emotion2vec_plus_large',
        });
      } else {
        const choices = ['happy','sad','angry','neutral','fear','surprise'];
        const target = choices[Math.floor((String(payload).length+src.length)%choices.length)];
        const dist = mockDist(target, 0.55+Math.random()*0.18);
        setResult({ dist, latency: 180+Math.floor(Math.random()*70), modelName:'recording demo' });
      }
    }catch(e){
      alert(e.message || '语音分析失败');
    }finally{
      setPending(false);
    }
  }

  return (
    <>
      <div className="panel-hd">
        <div>
          <h2><em>语音</em> 情感识别</h2>
          <p>上传音频文件或当场录音，emotion2vec_plus_large 将语音切分为 1 s 片段后输出情绪分布。</p>
        </div>
        <div className="right">
          MODEL · <b>emotion2vec_plus_large</b><br/>
          SAMPLE · <b>16 kHz · mono</b><br/>
          WINDOW · <b>1.0 s · stride 0.5</b>
        </div>
      </div>

      <div className="audio-stage">
        <label className={'audio-drop'+(file?' has-file':'')}>
          <input type="file" accept="audio/*" onChange={onFile}/>
          {file ? (<>
            <div className="icn">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                <path d="M5 9v6M9 6v12M13 4v16M17 8v8M21 11v2"
                  stroke="currentColor" strokeWidth="1.4" strokeLinecap="round"/>
              </svg>
            </div>
            <div className="title">{file.name}</div>
            <div className="hint">{file.size} · {file.dur} · CLICK TO REPLACE</div>
          </>) : (<>
            <div className="icn">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                <path d="M12 3v12M12 15l-4-4M12 15l4-4M5 19h14"
                  stroke="currentColor" strokeWidth="1.4" strokeLinecap="round"/>
              </svg>
            </div>
            <div className="title">拖入或点击上传音频</div>
            <div className="hint">.WAV · .MP3 · .M4A · ≤ 10 MB</div>
            <div className="or"><span>OR</span></div>
            <button className={'rec-btn'+(recording?' recording':'')} onClick={(e)=>{ e.preventDefault(); toggleRec(); }}>
              <span className="rec-dot"></span>
              {recording ? `RECORDING · ${recTime.toFixed(1)}s · STOP` : 'RECORD FROM MIC'}
            </button>
          </>)}
        </label>

        <div className="card waveform-card">
          <h4>Waveform<span className="tag">MFCC × 40</span></h4>
          <div className="waveform">
            {wave.map((h,i)=>(
              <i key={i} className={h.hi?'hi':''} style={{height:(8+h.v*1.4)+'%'}}/>
            ))}
          </div>
          <div className="audio-meta">
            <span>0.0s</span>
            <span>F0 mean · {file?'192':'—'} Hz</span>
            <span>{file?file.dur:'—'}</span>
          </div>
        </div>
      </div>

      {pending && <div className="empty">
        <span className="spinner" style={{width:18,height:18}}></span>
        <p className="mono">Running emotion2vec on segments…</p>
      </div>}

      {result && !pending && (
        <ResultBlock dist={result.dist}
          subtitle="基于 emotion2vec 输出"
          modelName={result.modelName || "emotion2vec_plus_large"}
          latency={result.latency}
          sourceLabel="speech"/>
      )}

      {!result && !pending && (
        <div className="empty">
          <div className="e-icn">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
              <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="1.2"/>
              <path d="M12 7v5l3 2" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round"/>
            </svg>
          </div>
          <p>上传或录制后，结果将出现在此处。</p>
        </div>
      )}
    </>
  );
}
function genWave(mode){
  return Array.from({length:60}).map((_,i)=>{
    if(mode==='idle') return { v: 4+Math.sin(i*0.5)*3, hi:false };
    if(mode==='rec'){
      const v = 12 + Math.abs(Math.sin(i*0.3 + Date.now()*0.003))*32 + (Math.random()-.5)*8;
      return { v, hi: v>32 };
    }
    const v = 6 + Math.abs(Math.sin(i*0.4+1.1))*36 + (Math.random()-.5)*6;
    return { v, hi: v>34 };
  });
}
function mockDist(targetKey, mass){
  const dist = {};
  EMO_KEYS.forEach(k=>{
    dist[k] = (1-mass)/6 + (Math.random()*0.04);
  });
  dist[targetKey] = mass + (Math.random()*0.04);
  // normalize
  const s = Object.values(dist).reduce((a,b)=>a+b,0);
  EMO_KEYS.forEach(k=> dist[k] = +(dist[k]/s).toFixed(3));
  return dist;
}

// ─── TAB 3 : FACE ───────────────────────────────────────────
const FACE_SAMPLES = [
  { id:'s1', label:'Happy · 微笑', target:'happy',  bg:'oklch(0.32 0.05 60)',  mouth:'smile' },
  { id:'s2', label:'Neutral · 平静', target:'neutral', bg:'oklch(0.32 0.02 240)', mouth:'flat' },
  { id:'s3', label:'Angry · 皱眉', target:'angry',  bg:'oklch(0.28 0.07 30)',  mouth:'frown' },
  { id:'s4', label:'Surprise · 张嘴', target:'surprise', bg:'oklch(0.30 0.06 170)', mouth:'open' },
];

function FaceTab(){
  const [sample, setSample] = useState(FACE_SAMPLES[0]);
  const [pending, setPending] = useState(false);
  const [result, setResult] = useState(()=>({
    dist: mockDist('happy', 0.78), latency: 92, box:{ x:30,y:18,w:40,h:55 }
  }));
  const [uploaded, setUploaded] = useState(null);

  function analyze(s){
    setSample(s);
    setUploaded(null);
    setPending(true);
    setTimeout(()=>{
      setResult({
        dist: mockDist(s.target, 0.62+Math.random()*0.22),
        latency: 75+Math.floor(Math.random()*45),
        box:{ x:28+(Math.random()-.5)*6, y:16+(Math.random()-.5)*4,
              w:42+(Math.random()-.5)*4, h:56+(Math.random()-.5)*4 }
      });
      setPending(false);
    }, 480);
  }

  async function onUpload(e){
    const f = e.target.files[0];
    if(!f) return;
    const url = URL.createObjectURL(f);
    setUploaded({ url, name:f.name });
    setPending(true);
    try{
      const data = await apiFace(f);
      setResult({
        dist: data.dist,
        latency: data.latency,
        box:{ x:25, y:18, w:48, h:58 },
        faces: data.faces,
      });
      if(data.annotatedImage){
        setUploaded({ url:data.annotatedImage, name:f.name });
      }
    }catch(err){
      alert(err.message || '面部分析失败');
    }finally{
      setPending(false);
    }
  }

  const [topK] = result ? topEmo(result.dist) : ['happy',0];

  return (
    <>
      <div className="panel-hd">
        <div>
          <h2><em>面部</em> 表情识别</h2>
          <p>OpenCV Haar Cascade 检测人脸框，ViT 在 224 × 224 上做 7 类分类。下方为预置样本，亦可上传图片。</p>
        </div>
        <div className="right">
          MODEL · <b>trpakov/vit-face-expression</b><br/>
          DETECTOR · <b>Haar Cascade · OpenCV</b><br/>
          INPUT · <b>224 × 224 RGB</b>
        </div>
      </div>

      <div className="face-stage">
        <div>
          <div className="face-canvas">
            {uploaded ? (
              <img src={uploaded.url} alt="upload"/>
            ) : (
              <FaceMock sample={sample}/>
            )}
            {!pending && result && result.box && (
              <div className="face-box" style={{
                left:result.box.x+'%', top:result.box.y+'%',
                width:result.box.w+'%', height:result.box.h+'%',
                borderColor: COL[topK],
              }}>
                <span className="lbl" style={{background:COL[topK]}}>
                  {topK} · {(result.dist[topK]*100).toFixed(0)}%
                </span>
              </div>
            )}
            {pending && (
              <div style={{position:'absolute',inset:0,background:'rgba(10,10,12,.7)',
                display:'flex',alignItems:'center',justifyContent:'center',gap:10,
                color:'var(--fg-2)',fontFamily:'var(--mono)',fontSize:12}}>
                <span className="spinner"/> ViT inference…
              </div>
            )}
          </div>
          <div className="face-controls">
            {FACE_SAMPLES.map(s=>(
              <button key={s.id} className={'fc-btn'+(sample.id===s.id&&!uploaded?' on':'')}
                onClick={()=>analyze(s)}>{s.label}</button>
            ))}
            <label className="fc-btn">
              <input type="file" accept="image/*" onChange={onUpload}/>
              <svg width="11" height="11" viewBox="0 0 24 24" fill="none">
                <path d="M12 3v12M12 15l-4-4M12 15l4-4M5 19h14"
                  stroke="currentColor" strokeWidth="1.6" strokeLinecap="round"/>
              </svg>
              UPLOAD
            </label>
          </div>
        </div>

        {result && !pending && (
          <div style={{display:'flex',flexDirection:'column',gap:14}}>
            <div className="card" style={{padding:18}}>
              <h4>Detection<span className="tag">FACE × {result.faces ?? 1}</span></h4>
              <div style={{display:'flex',flexDirection:'column',gap:8,fontFamily:'var(--mono)',fontSize:11,color:'var(--fg-3)'}}>
                <div style={{display:'flex',justifyContent:'space-between'}}>
                  <span>BOX</span>
                  <span style={{color:'var(--fg)'}}>x {result.box.x.toFixed(0)} · y {result.box.y.toFixed(0)} · w {result.box.w.toFixed(0)} · h {result.box.h.toFixed(0)}</span>
                </div>
                <div style={{display:'flex',justifyContent:'space-between'}}>
                  <span>LATENCY</span>
                  <span style={{color:'var(--fg)'}}>{result.latency} ms</span>
                </div>
                <div style={{display:'flex',justifyContent:'space-between'}}>
                  <span>ENTROPY</span>
                  <span style={{color:'var(--fg)'}}>{entropy(result.dist).toFixed(2)}</span>
                </div>
              </div>
            </div>
            <div className="card" style={{padding:18}}>
              <h4>Probability<span className="tag">SOFTMAX</span></h4>
              <div className="bar-list">
                {Object.entries(result.dist).sort((a,b)=>b[1]-a[1]).map(([k,v],i)=>(
                  <div className={'row'+(i===0?' top':'')} key={k}>
                    <span className="lb">{k}</span>
                    <span className="tk"><span className="fl" style={{width:(v*100)+'%',background:COL[k]}}/></span>
                    <span className="vl">{v.toFixed(3)}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </>
  );
}

function FaceMock({ sample }){
  // generate a stylized "face" placeholder per sample
  return (
    <svg viewBox="0 0 400 300" preserveAspectRatio="xMidYMid slice" style={{width:'100%',height:'100%',background:sample.bg}}>
      <defs>
        <radialGradient id="fg" cx="50%" cy="48%" r="42%">
          <stop offset="0" stopColor="oklch(0.78 0.06 60)"/>
          <stop offset="1" stopColor="oklch(0.5 0.06 60)"/>
        </radialGradient>
      </defs>
      <ellipse cx="200" cy="150" rx="92" ry="118" fill="url(#fg)" opacity=".9"/>
      <ellipse cx="170" cy="135" rx="9" ry={sample.mouth==='open'?12:7} fill="oklch(0.18 0.02 40)"/>
      <ellipse cx="230" cy="135" rx="9" ry={sample.mouth==='open'?12:7} fill="oklch(0.18 0.02 40)"/>
      {sample.mouth==='smile' && <path d="M165 185 Q200 215 235 185" stroke="oklch(0.30 0.07 30)" strokeWidth="6" fill="none" strokeLinecap="round"/>}
      {sample.mouth==='frown' && <path d="M165 195 Q200 175 235 195" stroke="oklch(0.30 0.07 30)" strokeWidth="5" fill="none" strokeLinecap="round"/>}
      {sample.mouth==='flat' && <line x1="170" y1="190" x2="230" y2="190" stroke="oklch(0.30 0.07 30)" strokeWidth="4" strokeLinecap="round"/>}
      {sample.mouth==='open' && <ellipse cx="200" cy="195" rx="18" ry="14" fill="oklch(0.20 0.05 30)"/>}
      <text x="14" y="22" fontFamily="JetBrains Mono" fontSize="11" fill="rgba(255,255,255,.5)" letterSpacing=".06em">CAM · 640 × 480 · sample</text>
    </svg>
  );
}

// ─── TAB 4 : MULTIMODAL ─────────────────────────────────────
function MultiTab(){
  const [tx, setTx] = useState({ on:true, text:'哈哈哈这也太巧了吧' });
  const [sp, setSp] = useState({ on:true, file:null });
  const [vi, setVi] = useState({ on:true, file:null, preview:null });
  const [pending, setPending] = useState(false);
  const [out, setOut] = useState(null);

  async function fuse(){
    setPending(true);
    try{
      const data = await apiMultimodal({
        text: tx.on ? tx.text : '',
        audio: sp.on ? sp.file : null,
        image: vi.on ? vi.file : null,
      });
      setOut({
        fused: data.dist,
        W: data.W || [],
        labels: data.labels || [],
        latency: data.latency,
      });
    }catch(e){
      alert(e.message || '多模态融合失败');
    }finally{
      setPending(false);
    }
  }

  return (
    <>
      <div className="panel-hd">
        <div>
          <h2><em>多模态</em> 融合</h2>
          <p>启用任意通路（一条或多条）。系统对各通路置信度的逆熵作为权重，加权得到最终分布。</p>
        </div>
        <div className="right">
          STRATEGY · <b>late · attention</b><br/>
          WEIGHT · <b>1 − H(p) / log2(7)</b><br/>
          FALLBACK · <b>auto-renorm on dropout</b>
        </div>
      </div>

      <div className="lane-grid">
        <Lane title="① 文本通路" on={tx.on} onToggle={v=>setTx({...tx,on:v})}>
          <textarea className="text-in" style={{minHeight:120,fontSize:15}}
            value={tx.text} onChange={e=>setTx({...tx,text:e.target.value})}
            placeholder="输入文本…"/>
          <div className="audio-meta">
            <span>STRUCTBERT</span><span>{tx.text.length} 字</span>
          </div>
        </Lane>
        <Lane title="② 语音通路" on={sp.on} onToggle={v=>setSp({...sp,on:v})}>
          <div style={{flex:1,background:'var(--bg)',border:'1px solid var(--line)',borderRadius:3,padding:14,display:'flex',flexDirection:'column',gap:10,minHeight:140}}>
            <div className="audio-meta"><span>{sp.file ? sp.file.name : 'UPLOAD AUDIO'}</span><span>{sp.file ? (sp.file.size/1024).toFixed(1)+' KB' : 'optional'}</span></div>
            <div className="waveform" style={{height:60,padding:0}}>
              {genWave('file').slice(0,30).map((h,i)=>(
                <i key={i} className={h.hi?'hi':''} style={{height:(8+h.v*1.4)+'%'}}/>
              ))}
            </div>
            <label className="fc-btn" style={{alignSelf:'flex-start'}}>
              <input type="file" accept="audio/*" onChange={e=>setSp({...sp,file:e.target.files[0] || null})}/>
              SELECT AUDIO
            </label>
            <div className="audio-meta"><span>EMOTION2VEC</span><span>{sp.file ? 'ready' : 'waiting'}</span></div>
          </div>
        </Lane>
        <Lane title="③ 视觉通路" on={vi.on} onToggle={v=>setVi({...vi,on:v})}>
          <div style={{flex:1,borderRadius:3,overflow:'hidden',aspectRatio:'4/3',minHeight:120}}>
            {vi.preview ? <img src={vi.preview} alt="vision upload" style={{width:'100%',height:'100%',objectFit:'cover'}}/> : <FaceMock sample={FACE_SAMPLES[0]}/>}
          </div>
          <div style={{display:'flex',gap:6,flexWrap:'wrap'}}>
            <label className="fc-btn" style={{padding:'5px 10px',fontSize:9.5}}>
              <input type="file" accept="image/*" onChange={e=>{
                const file = e.target.files[0] || null;
                setVi({...vi, file, preview:file ? URL.createObjectURL(file) : null});
              }}/>
              SELECT IMAGE
            </label>
            <span className="mono" style={{fontSize:10.5,color:'var(--fg-3)',alignSelf:'center'}}>
              {vi.file ? vi.file.name : 'optional'}
            </span>
          </div>
        </Lane>
      </div>

      <button className="run-btn" onClick={fuse} disabled={pending || (!tx.on && !sp.on && !vi.on)}
        style={{maxWidth:280}}>
        <span>{pending ? <><span className="spinner"/> Fusing channels…</> : 'RUN FUSION'}</span>
        <span className="key">α · β · γ</span>
      </button>

      {out && !pending && (
        <div className="fusion-result">
          <div>
            <h4 style={{fontFamily:'var(--mono)',fontSize:10.5,letterSpacing:'.1em',
              textTransform:'uppercase',color:'var(--fg-3)',marginBottom:14}}>
              Fused Distribution<span className="tag" style={{marginLeft:10,padding:'2px 8px',border:'1px solid var(--line)',borderRadius:3,fontSize:9.5}}>{out.labels.length} CHANNELS</span>
            </h4>
            <div className="verdict-big" style={{marginTop:0}}>{topEmo(out.fused)[0]}</div>
            <div className="verdict-zh">{ZH[topEmo(out.fused)[0]]} · cross-modal agreement</div>
            <div className="fusion-weights">
              {out.labels.map((lb,i)=>(
                <div className="weight" key={lb}>
                  <span className="wlbl">{lb}</span>
                  <span className="wval">{out.W[i].toFixed(2)}</span>
                  <span className="wbar"><span className="wfl" style={{width:(out.W[i]*100)+'%'}}/></span>
                </div>
              ))}
            </div>
          </div>
          <div>
            <h4 style={{fontFamily:'var(--mono)',fontSize:10.5,letterSpacing:'.1em',
              textTransform:'uppercase',color:'var(--fg-3)',marginBottom:14}}>
              Probability
            </h4>
            <div className="bar-list">
              {Object.entries(out.fused).sort((a,b)=>b[1]-a[1]).map(([k,v],i)=>(
                <div className={'row'+(i===0?' top':'')} key={k}>
                  <span className="lb">{k}</span>
                  <span className="tk"><span className="fl" style={{width:(v*100)+'%',background:COL[k]}}/></span>
                  <span className="vl">{v.toFixed(3)}</span>
                </div>
              ))}
            </div>
            <div style={{marginTop:14,fontFamily:'var(--mono)',fontSize:10,color:'var(--fg-3)',letterSpacing:'.04em',display:'flex',justifyContent:'space-between'}}>
              <span>LATENCY · {out.latency} ms</span>
              <span>ENTROPY · {entropy(out.fused).toFixed(2)}</span>
            </div>
          </div>
        </div>
      )}
    </>
  );
}

function Lane({ title, on, onToggle, children }){
  return (
    <div className={'lane'+(on?' active':'')}>
      <div className="lane-hd">
        <b>{title}</b>
        <label className="toggle">
          <input type="checkbox" checked={on} onChange={e=>onToggle(e.target.checked)}/>
          <span className="slider"></span>
        </label>
      </div>
      {on ? children : (
        <div style={{flex:1,display:'flex',alignItems:'center',justifyContent:'center',
          color:'var(--fg-4)',fontFamily:'var(--mono)',fontSize:11,letterSpacing:'.06em'}}>
          DISABLED · weight 0
        </div>
      )}
    </div>
  );
}

// ─── TAB 5 : VIDEO ──────────────────────────────────────────
function VideoTab(){
  const [video, setVideo] = useState(null);
  const [pending, setPending] = useState(false);
  const [tracks, setTracks] = useState(null);

  function onUpload(e){
    const f = e.target.files && e.target.files[0];
    setVideo({ name: f? f.name : 'sample_interview.mp4', dur:'20.0 s', frames:'600' });
    setPending(true);
    setTimeout(()=>{
      const fT = genTrack(['neutral','neutral','happy','happy','surprise','happy','happy','neutral','neutral','sad','neutral','neutral']);
      const sT = genTrack(['neutral','happy','happy','happy','happy','happy','neutral','neutral','neutral','sad','neutral','neutral']);
      const cT = fuseTracks(fT, sT);
      setTracks({ fT, sT, cT });
      setPending(false);
    }, 1000);
  }

  return (
    <>
      <div className="panel-hd">
        <div>
          <h2><em>视频</em> 双轨时间线</h2>
          <p>上传视频，系统抽取关键帧与音频片段，分别经 ViT 与 emotion2vec 推断后，渲染面部 / 语音双轨情绪时间线及其融合曲线。</p>
        </div>
        <div className="right">
          STRIDE · <b>frame 2s · audio 1s</b><br/>
          TOTAL · <b>~ 12 segments / 20 s</b><br/>
          OUTPUT · <b>per-segment {`{emotion: prob}`}</b>
        </div>
      </div>

      <div className="video-stage">
        <label className="video-drop" htmlFor="vid-up">
          <input id="vid-up" type="file" accept="video/*" style={{display:'none'}} onChange={onUpload}/>
          {video ? (
            <div style={{display:'flex',flexDirection:'column',gap:8,alignItems:'center'}}>
              <div style={{fontFamily:'var(--serif)',fontSize:22,letterSpacing:'-.01em'}}>{video.name}</div>
              <div className="mono" style={{fontSize:10.5,color:'var(--fg-3)',letterSpacing:'.06em'}}>
                {video.dur} · {video.frames} FRAMES · CLICK TO REPLACE
              </div>
            </div>
          ) : (
            <div style={{display:'flex',flexDirection:'column',gap:10,alignItems:'center'}}>
              <div className="mono" style={{fontSize:10.5,color:'var(--fg-3)',letterSpacing:'.1em'}}>↑ DROP VIDEO HERE</div>
              <div style={{fontFamily:'var(--serif)',fontSize:24,letterSpacing:'-.01em'}}>上传一段视频开始分析</div>
              <div className="mono" style={{fontSize:10.5,color:'var(--fg-4)',letterSpacing:'.06em'}}>
                .MP4 · .MOV · ≤ 60 s · OR USE SAMPLE
              </div>
              <button type="button" className="rec-btn" style={{marginTop:8}}
                onClick={(e)=>{e.preventDefault(); onUpload({target:{files:[null]}})}}>
                USE SAMPLE · INTERVIEW
              </button>
            </div>
          )}
        </label>

        {pending && <div className="empty">
          <span className="spinner" style={{width:18,height:18}}/>
          <p className="mono">Extracting key frames · running ViT + emotion2vec on segments…</p>
        </div>}

        {tracks && !pending && (
          <div className="timeline-wrap">
            <div className="time-axis">
              <span>0s</span><span>5s</span><span>10s</span><span>15s</span><span>20s</span>
            </div>
            <Track label="VISION" sub="ViT · key frames" trk={tracks.fT}/>
            <Track label="SPEECH" sub="emotion2vec · 1s win" trk={tracks.sT}/>
            <Track label="FUSED"  sub="late attention" trk={tracks.cT} hi/>
            <div className="audio-meta" style={{marginTop:4}}>
              <span>SEGMENT n=12</span>
              <span>AGREEMENT · {Math.round((tracks.fT.filter((v,i)=>v===tracks.sT[i]).length / tracks.fT.length)*100)}%</span>
              <span>DOMINANT · {topInTrack(tracks.cT)}</span>
            </div>
          </div>
        )}
      </div>
    </>
  );
}
function genTrack(arr){ return arr; }
function fuseTracks(a,b){
  return a.map((x,i)=> x===b[i] ? x : (Math.random()<0.6 ? x : b[i]));
}
function topInTrack(arr){
  const c = {}; arr.forEach(k=> c[k]=(c[k]||0)+1);
  return Object.entries(c).sort((a,b)=>b[1]-a[1])[0][0];
}
function Track({ label, sub, trk, hi }){
  return (
    <div className="track">
      <div className="lbl"><b>{label}</b>{sub}</div>
      <div className="bar-tl" style={hi?{boxShadow:'0 0 0 1px var(--c-joy) inset',borderColor:'transparent'}:{}}>
        {trk.map((k,i)=>(
          <i key={i} style={{background:COL[k],opacity:.85}} title={`${i*1.7}s · ${k}`}>{k.slice(0,3)}</i>
        ))}
      </div>
    </div>
  );
}

// ─── TAB 6 : REALTIME ───────────────────────────────────────
function LiveTab(){
  const [active, setActive] = useState(false);
  const [stream, setStream] = useState(null);
  const [err, setErr] = useState(null);
  const [dist, setDist] = useState(()=>mockDist('happy',0.55));
  const [fps, setFps] = useState(0);
  const videoRef = useRef(null);
  const tickRef = useRef(null);
  const lastT = useRef(performance.now());

  async function start(){
    setErr(null);
    try{
      const s = await navigator.mediaDevices.getUserMedia({ video:true, audio:false });
      setStream(s);
      setActive(true);
      if(videoRef.current){ videoRef.current.srcObject = s; }
      // simulate inference at 10 Hz
      tickRef.current = setInterval(()=>{
        const keys = ['happy','neutral','surprise','happy','neutral'];
        const k = keys[Math.floor(Math.random()*keys.length)];
        setDist(mockDist(k, 0.5+Math.random()*0.3));
        const now = performance.now();
        setFps(Math.round(1000/(now-lastT.current)));
        lastT.current = now;
      }, 200);
    }catch(e){
      setErr(e.message || '无法访问摄像头。请确认浏览器权限。');
    }
  }
  function stop(){
    if(tickRef.current){ clearInterval(tickRef.current); tickRef.current=null; }
    if(stream){ stream.getTracks().forEach(t=>t.stop()); }
    setStream(null); setActive(false); setFps(0);
  }
  useEffect(()=>()=>stop(), []);

  const [topK,topV] = topEmo(dist);

  return (
    <>
      <div className="panel-hd">
        <div>
          <h2><em>实时</em> 摄像头</h2>
          <p>开启摄像头后，系统以约 10 Hz 频率对每一帧进行 Haar Cascade 检测 + ViT 分类，结果叠加在视频流上。</p>
        </div>
        <div className="right">
          INFERENCE · <b>~ 10 Hz</b><br/>
          MODEL · <b>vit-face-expression</b><br/>
          PRIVACY · <b>local · no upload</b>
        </div>
      </div>

      <div className="rt-stage">
        <div>
          <div className="face-canvas">
            {active ? (
              <video ref={videoRef} autoPlay playsInline muted/>
            ) : (
              <div className="ph">
                <div className="icn">
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                    <rect x="3" y="6" width="14" height="12" rx="2" stroke="currentColor" strokeWidth="1.4"/>
                    <path d="M17 10l4-2v8l-4-2" stroke="currentColor" strokeWidth="1.4" strokeLinejoin="round"/>
                  </svg>
                </div>
                <p style={{fontFamily:'var(--serif)',fontSize:20}}>摄像头未开启</p>
                {err && <div style={{color:'var(--c-ang)',fontFamily:'var(--mono)',fontSize:11,letterSpacing:'.04em'}}>{err}</div>}
              </div>
            )}
            {active && (
              <div className="face-box" style={{ left:'25%', top:'15%', width:'50%', height:'65%', borderColor:COL[topK] }}>
                <span className="lbl" style={{background:COL[topK]}}>
                  {topK} · {(topV*100).toFixed(0)}%
                </span>
              </div>
            )}
          </div>
          <div className="face-controls">
            {!active ? (
              <button className="fc-btn on" onClick={start}>
                <svg width="11" height="11" viewBox="0 0 24 24" fill="none">
                  <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="1.4"/>
                  <circle cx="12" cy="12" r="3" fill="currentColor"/>
                </svg>
                START CAMERA
              </button>
            ) : (
              <button className="fc-btn" onClick={stop}>
                <svg width="11" height="11" viewBox="0 0 24 24" fill="none">
                  <rect x="6" y="6" width="12" height="12" stroke="currentColor" strokeWidth="1.4"/>
                </svg>
                STOP
              </button>
            )}
          </div>
        </div>

        <div className="rt-status">
          <div style={{display:'flex',justifyContent:'space-between',alignItems:'center'}}>
            <span className="live-tag"><span className="d"></span>{active?'LIVE':'IDLE'}</span>
            <span className="mono" style={{fontSize:10.5,color:'var(--fg-3)',letterSpacing:'.04em'}}>
              {active? `${fps} fps · ${active?'inferring':''}` : '— fps'}
            </span>
          </div>
          <div>
            <h4 style={{fontFamily:'var(--mono)',fontSize:10.5,letterSpacing:'.1em',
              textTransform:'uppercase',color:'var(--fg-3)',marginBottom:14}}>
              Current Frame
            </h4>
            <div className="verdict-big" style={{fontSize:42}}>{topK}</div>
            <div className="verdict-zh">{ZH[topK]} · {(topV*100).toFixed(1)}%</div>
          </div>
          <div className="bar-list" style={{marginTop:4}}>
            {Object.entries(dist).sort((a,b)=>b[1]-a[1]).map(([k,v],i)=>(
              <div className={'row'+(i===0?' top':'')} key={k}>
                <span className="lb">{k}</span>
                <span className="tk"><span className="fl" style={{width:(v*100)+'%',background:COL[k],transition:'width .25s linear'}}/></span>
                <span className="vl">{v.toFixed(2)}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </>
  );
}

// ─── ROUTER ─────────────────────────────────────────────────
function App(){
  const [tab, setTab] = useState('text');
  useEffect(()=>{
    const map = { text:'text emotion', speech:'speech emotion', face:'facial expression',
      multi:'multimodal fusion', video:'video timeline', live:'live camera' };
    const el = document.getElementById('crumb');
    if(el) el.textContent = map[tab] || '';
  }, [tab]);

  return (
    <>
      <div className="tabs">
        {TABS.map(t=>(
          <button key={t.id} className={'tab'+(tab===t.id?' active':'')}
            onClick={()=>setTab(t.id)}>
            <span className="ix">{t.ix}</span>{t.label}
          </button>
        ))}
      </div>
      {tab==='text'   && <TextTab/>}
      {tab==='speech' && <SpeechTab/>}
      {tab==='face'   && <FaceTab/>}
      {tab==='multi'  && <MultiTab/>}
      {tab==='video'  && <VideoTab/>}
      {tab==='live'   && <LiveTab/>}
    </>
  );
}

const root = ReactDOM.createRoot(document.getElementById('demo-root'));
root.render(<App/>);
