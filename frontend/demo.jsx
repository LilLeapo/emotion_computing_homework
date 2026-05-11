// Demo widget for the showcase: preset multimodal cases + animated result bars.

const { useState, useEffect, useMemo } = React;

const CASES = [
  {
    id: 'c1',
    title: '社交聚会',
    sub: 'webcam · mic · 实时',
    head: 'A1 · spontaneous · 04.2 s',
    face: { activation: 'mouth+eyes', confidence: 0.88, label: 'smile' },
    audio: {
      bars: [3,5,8,12,18,22,28,34,38,42,40,36,32,28,24,22,28,34,40,44,42,36,30,24,20,18,22,26,30,28,24,20,16,14,12,10,8,6,5,4],
      hi: [4,5,6,7,8,9,10,11,12,13,17,18,19,20,21],
      label: 'rising pitch · laughter',
      confidence: 0.81,
    },
    text: {
      quote: '"哈哈哈这也太巧了吧，我们居然又见面了。"',
      hi: ['哈哈哈','太巧了','又见面了'],
      label: 'positive sentiment',
      confidence: 0.86,
    },
    weights: { v:0.41, s:0.32, t:0.27 },
    dist: { happy:0.612, surprise:0.183, neutral:0.094, sad:0.046, angry:0.031, fear:0.022, disgust:0.012 },
    verdict: 'happy',
    notes: '三模态一致指向喜悦类。视觉端检测到唇角上扬与眼轮匝肌收缩（Duchenne 标记），语音端基频骤升伴随笑声爆破，文本端拟声词与积极副词共现。',
  },
  {
    id: 'c2',
    title: '负面反馈',
    sub: '工位 · 邮件读出',
    head: 'A2 · prepared · 06.8 s',
    face: { activation: 'brow+nose', confidence: 0.74, label: 'frown' },
    audio: {
      bars: [22,24,28,32,30,28,32,36,40,38,32,28,26,30,34,32,28,24,22,26,30,28,24,20,18,22,28,30,26,22,20,18,16,18,20,18,16,14,12,10],
      hi: [7,8,9,10,11,15,16,17],
      label: 'low pitch · sustained',
      confidence: 0.79,
    },
    text: {
      quote: '"这个版本又退化了，根本没法用。"',
      hi: ['又退化了','根本没法用'],
      label: 'negative · frustration',
      confidence: 0.83,
    },
    weights: { v:0.28, s:0.34, t:0.38 },
    dist: { angry:0.484, disgust:0.226, sad:0.118, neutral:0.082, fear:0.054, surprise:0.024, happy:0.012 },
    verdict: 'angry',
    notes: '文本主导：极性副词与否定结构置信度最高。语音呈现持续低音量低基频；视觉端皱眉肌活动支撑，但角度受限导致权重略低。',
  },
  {
    id: 'c3',
    title: '陈述事实',
    sub: 'meeting · 静态',
    head: 'A3 · neutral baseline · 03.5 s',
    face: { activation: 'flat', confidence: 0.69, label: 'neutral' },
    audio: {
      bars: [12,14,16,18,18,16,18,20,22,20,18,16,18,20,22,20,18,16,14,16,18,20,18,16,14,16,18,20,22,20,16,14,12,14,16,14,12,10,8,8],
      hi: [],
      label: 'even pace',
      confidence: 0.72,
    },
    text: {
      quote: '"会议安排在下周三下午三点。"',
      hi: ['下周三','下午三点'],
      label: 'informational',
      confidence: 0.78,
    },
    weights: { v:0.31, s:0.31, t:0.38 },
    dist: { neutral:0.692, happy:0.092, sad:0.084, surprise:0.058, angry:0.034, fear:0.024, disgust:0.016 },
    verdict: 'neutral',
    notes: '三通路一致呈现低唤起特征。事实性陈述、稳定基频与表情平淡形成相互佐证，无显著情感激活。',
  },
  {
    id: 'c4',
    title: '紧张陈述',
    sub: '答辩现场',
    head: 'A4 · high arousal · 05.1 s',
    face: { activation: 'eyes+brow', confidence: 0.71, label: 'tense' },
    audio: {
      bars: [8,12,18,24,32,38,42,40,36,30,32,38,44,46,42,36,28,22,24,30,38,42,38,30,24,20,22,28,34,38,32,26,22,18,16,14,12,10,8,6],
      hi: [4,5,6,7,11,12,13,14,20,21,22,23,28,29,30],
      label: 'unsteady · tremor',
      confidence: 0.77,
    },
    text: {
      quote: '"那个 …… 我觉得，我刚才说的可能不太对。"',
      hi: ['那个','可能不太对'],
      label: 'hedging · uncertainty',
      confidence: 0.74,
    },
    weights: { v:0.36, s:0.40, t:0.24 },
    dist: { fear:0.412, sad:0.198, surprise:0.142, neutral:0.108, angry:0.064, disgust:0.046, happy:0.030 },
    verdict: 'fear',
    notes: '语音颤抖与停顿密度为主导信号，视觉端眨眼频率显著升高。文本含犹疑标记，三通路加权后定位为恐惧/紧张子类。',
  },
];

const PALETTE = {
  happy:    'oklch(0.82 0.13 82)',
  surprise: 'oklch(0.78 0.13 165)',
  neutral:  'oklch(0.62 0.01 80)',
  sad:      'oklch(0.65 0.10 230)',
  angry:    'oklch(0.62 0.16 25)',
  fear:     'oklch(0.55 0.09 290)',
  disgust:  'oklch(0.58 0.08 130)',
};

const ZH = {
  happy:'喜悦', surprise:'惊讶', neutral:'中性',
  sad:'悲伤', angry:'愤怒', fear:'恐惧', disgust:'厌恶',
};

function Bars({ dist, verdict }) {
  // sort by value desc
  const sorted = Object.entries(dist).sort((a,b)=>b[1]-a[1]);
  return (
    <div className="bars">
      {sorted.map(([k,v]) => (
        <div className={'bar-row' + (k===verdict?' top':'')} key={k}>
          <div className="lbl">{k} · {ZH[k]}</div>
          <div className="track">
            <div className="fill" style={{ width:(v*100).toFixed(1)+'%', background:PALETTE[k] }}/>
          </div>
          <div className="val">{v.toFixed(3)}</div>
        </div>
      ))}
    </div>
  );
}

function Demo() {
  const [idx, setIdx] = useState(0);
  const c = CASES[idx];
  const conf = c.dist[c.verdict];

  // Reset transition when case changes
  const [animKey, setAnimKey] = useState(0);
  useEffect(() => { setAnimKey(k => k + 1); }, [idx]);

  return (
    <div className="demo">
      {/* case picker */}
      <aside className="demo-cases">
        {CASES.map((cs, i) => (
          <div key={cs.id}
               className={'case-row' + (i===idx?' active':'')}
               onClick={() => setIdx(i)}>
            <div className="ix">{String(i+1).padStart(2,'0')}</div>
            <div className="info">
              <b>{cs.title}</b>
              <span>{cs.sub}</span>
            </div>
          </div>
        ))}
      </aside>

      {/* stage */}
      <div className="demo-stage" key={animKey}>
        <div className="demo-head">
          <div className="l">
            <h3>Case {idx+1} · {c.title}</h3>
            <p>{c.head} · α {c.weights.v.toFixed(2)} · β {c.weights.s.toFixed(2)} · γ {c.weights.t.toFixed(2)}</p>
          </div>
          <div className="r">
            <div>SAMPLE RATE <b>16 kHz</b></div>
            <div>FRAME <b>30 fps</b></div>
            <div>LATENCY <b>~ 120 ms</b></div>
          </div>
        </div>

        {/* three modalities */}
        <div className="modality-grid">
          {/* face */}
          <div className="mview">
            <div className="mh">
              <span>① VISION · {c.face.label}</span>
              <span className="conf">{(c.face.confidence*100).toFixed(0)}%</span>
            </div>
            <div className="face-ph">
              <svg viewBox="0 0 200 160" preserveAspectRatio="xMidYMid slice">
                <ellipse cx="100" cy="82" rx="48" ry="62" fill="none" stroke="rgba(236,232,223,.35)" strokeWidth=".8"/>
                <circle cx="80" cy="72" r="4.5" fill="none" stroke="rgba(236,232,223,.55)" strokeWidth=".8"/>
                <circle cx="120" cy="72" r="4.5" fill="none" stroke="rgba(236,232,223,.55)" strokeWidth=".8"/>
                {c.face.label === 'smile' && (
                  <path d="M78 102 Q100 118 122 102" fill="none" stroke="rgb(241,202,118)" strokeWidth="1.4" strokeLinecap="round"/>
                )}
                {c.face.label === 'frown' && (
                  <path d="M78 108 Q100 96 122 108" fill="none" stroke="oklch(0.62 0.16 25)" strokeWidth="1.4" strokeLinecap="round"/>
                )}
                {c.face.label === 'neutral' && (
                  <path d="M82 104 L118 104" fill="none" stroke="rgba(236,232,223,.55)" strokeWidth="1.2" strokeLinecap="round"/>
                )}
                {c.face.label === 'tense' && (
                  <>
                    <path d="M82 104 Q100 98 118 104" fill="none" stroke="oklch(0.55 0.09 290)" strokeWidth="1.2" strokeLinecap="round"/>
                    <path d="M70 60 L88 64 M112 64 L130 60" stroke="oklch(0.55 0.09 290)" strokeWidth=".8"/>
                  </>
                )}
                {/* landmarks */}
                <g fill="rgba(241,202,118,.7)">
                  <circle cx="68" cy="78" r="1.2"/>
                  <circle cx="92" cy="78" r="1.2"/>
                  <circle cx="108" cy="78" r="1.2"/>
                  <circle cx="132" cy="78" r="1.2"/>
                  <circle cx="100" cy="92" r="1.2"/>
                  <circle cx="78" cy="110" r="1.2"/>
                  <circle cx="100" cy="116" r="1.2"/>
                  <circle cx="122" cy="110" r="1.2"/>
                </g>
              </svg>
              <div className="ph-cap">68 LANDMARKS · {c.face.activation.toUpperCase()}</div>
            </div>
          </div>

          {/* audio */}
          <div className="mview">
            <div className="mh">
              <span>② SPEECH · {c.audio.label}</span>
              <span className="conf">{(c.audio.confidence*100).toFixed(0)}%</span>
            </div>
            <div className="wave">
              {c.audio.bars.map((h, i) => (
                <i key={i}
                   className={c.audio.hi.includes(i)?'hi':''}
                   style={{ height: (8 + h*1.6) + '%' }}/>
              ))}
            </div>
            <div className="text-block" style={{flex:'0 0 auto'}}>
              <div className="meta">F0 mean · 192 Hz · jitter 1.4% · MFCC×40</div>
            </div>
          </div>

          {/* text */}
          <div className="mview">
            <div className="mh">
              <span>③ TEXT · {c.text.label}</span>
              <span className="conf">{(c.text.confidence*100).toFixed(0)}%</span>
            </div>
            <div className="text-block">
              <div className="quote">
                {c.text.quote.split('').map((ch, i) => {
                  const isHi = c.text.hi.some(h => {
                    const s = c.text.quote.indexOf(h);
                    return i >= s && i < s + h.length;
                  });
                  return isHi
                    ? <span className="word" key={i}>{ch}</span>
                    : <span key={i}>{ch}</span>;
                })}
              </div>
              <div className="meta">BERT-base-zh · 768 d → 256 d · attention-pooled</div>
            </div>
          </div>
        </div>

        {/* result */}
        <div className="result">
          <Bars dist={c.dist} verdict={c.verdict} />
          <div className="diag">
            <div className="diag-h">FUSED DIAGNOSIS</div>
            <div className="diag-verdict">
              <em>{c.verdict}</em>
              <span style={{fontFamily:'var(--mono)',fontSize:'13px',color:'var(--fg-3)'}}>
                {ZH[c.verdict]}
              </span>
            </div>
            <div className="diag-conf">
              <div className="ring-wrap">
                <div className="ring" style={{'--p': (conf*100).toFixed(1)}}></div>
                <span>{(conf*100).toFixed(0)}</span>
              </div>
              <span>融合置信度 · cross-modal agreement {Math.min(99,Math.round(conf*100+8))}</span>
            </div>
            <div className="diag-notes">
              <span className="ll">Rationale · 模型理由</span>
              {c.notes}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

const root = ReactDOM.createRoot(document.getElementById('demo-app'));
root.render(<Demo />);
