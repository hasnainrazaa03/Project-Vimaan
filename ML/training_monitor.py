"""Live, real-time training monitor for the Vimaan NLU trainer.

Serves a small dark dashboard that tails the metrics stream written by
``train_nlu_model.py`` (``ML/training_runs/<version>/metrics.jsonl``) and draws
live charts: the dataset split, the intent distribution, the loss curve
(per-step train loss with per-epoch train/val overlays), and the per-epoch
validation intent-accuracy and slot-F1.

Pure standard library (http.server) + vanilla-canvas charts — no extra pip
installs, no CDN, works fully offline. Run it in one terminal while training
runs in another::

    python ML/training_monitor.py            # newest run, http://localhost:8788
    python ML/training_monitor.py --run v11 --port 8788
"""

from __future__ import annotations

import argparse
import json
import os
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

RUNS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_runs")
MAX_STEP_POINTS = 800  # downsample the step curve so the payload stays small


def _newest_run(runs_dir: str) -> str | None:
    if not os.path.isdir(runs_dir):
        return None
    runs = [
        os.path.join(runs_dir, d)
        for d in os.listdir(runs_dir)
        if os.path.isfile(os.path.join(runs_dir, d, "metrics.jsonl"))
    ]
    if not runs:
        return None
    return max(runs, key=os.path.getmtime)


def _read_metrics(run_dir: str) -> dict:
    """Parse a run's metrics.jsonl into {meta, steps, epochs, done}."""
    path = os.path.join(run_dir, "metrics.jsonl")
    meta, done, steps, epochs = None, None, [], []
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue  # a half-written last line while the trainer flushes
                t = rec.get("type")
                if t == "meta":
                    meta = rec
                elif t == "step":
                    steps.append(rec)
                elif t == "epoch":
                    epochs.append(rec)
                elif t == "done":
                    done = rec
    except FileNotFoundError:
        pass

    # Downsample the step curve evenly, always keeping the last point.
    if len(steps) > MAX_STEP_POINTS:
        k = len(steps) / MAX_STEP_POINTS
        idx = sorted({int(i * k) for i in range(MAX_STEP_POINTS)} | {len(steps) - 1})
        steps = [steps[i] for i in idx]

    return {
        "run": os.path.basename(run_dir),
        "meta": meta,
        "steps": [{"step": s["step"], "loss": s["loss"], "epoch": s["epoch"]} for s in steps],
        "epochs": epochs,
        "done": done,
    }


class Handler(BaseHTTPRequestHandler):
    runs_dir = RUNS_DIR
    pinned_run: str | None = None

    def _send(self, code, body, ctype):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.split("?")[0] in ("/", "/index.html"):
            self._send(200, PAGE, "text/html; charset=utf-8")
            return
        if self.path.split("?")[0] == "/api/metrics":
            run_dir = (
                os.path.join(self.runs_dir, self.pinned_run)
                if self.pinned_run
                else _newest_run(self.runs_dir)
            )
            if not run_dir or not os.path.isdir(run_dir):
                self._send(200, json.dumps({"waiting": True}), "application/json")
                return
            self._send(200, json.dumps(_read_metrics(run_dir)), "application/json")
            return
        self._send(404, "not found", "text/plain")

    def log_message(self, *args):  # silence per-request stderr spam
        pass


class DualStackServer(ThreadingHTTPServer):
    """Listen on IPv6 with dual-stack so BOTH http://localhost (which many
    systems resolve to IPv6 ::1) and http://127.0.0.1 reach the monitor."""

    address_family = socket.AF_INET6
    allow_reuse_address = True

    def server_bind(self):
        try:
            self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
        except (AttributeError, OSError):
            pass
        super().server_bind()


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Vimaan Training Monitor</title>
<style>
  :root{
    --bg:#0b1220; --surface:#0e1626; --surface2:#121d31; --line:#20304d;
    --ink:#e7edf7; --muted:#8ea3c2; --faint:#5b6f92;
    --blue:#56b4e9; --orange:#e69f00; --green:#34d399; --purple:#a78bfa; --red:#f0776a;
    --good:#34d399;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);
    font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;}
  .wrap{max-width:1240px;margin:0 auto;padding:20px 20px 60px;}
  header.top{display:flex;flex-wrap:wrap;align-items:center;gap:10px 14px;
    padding:14px 16px;background:var(--surface);border:1px solid var(--line);border-radius:14px;}
  .title{font-size:16px;font-weight:700;letter-spacing:.2px;margin-right:4px}
  .title small{color:var(--muted);font-weight:500}
  .badge{font-size:12px;color:var(--muted);background:var(--surface2);
    border:1px solid var(--line);border-radius:999px;padding:4px 10px;white-space:nowrap}
  .badge b{color:var(--ink);font-weight:600}
  .pill{font-size:12px;font-weight:700;border-radius:999px;padding:4px 12px;letter-spacing:.3px}
  .pill.run{color:#04240f;background:var(--good)}
  .pill.wait{color:#0b1220;background:var(--muted)}
  .pill.done{color:#04240f;background:var(--blue)}
  .spacer{flex:1}
  .grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:16px}
  .card{background:var(--surface);border:1px solid var(--line);border-radius:14px;padding:14px 16px 16px}
  .card h2{margin:0 0 2px;font-size:13px;font-weight:700;letter-spacing:.3px;text-transform:uppercase;color:var(--muted)}
  .card .sub{font-size:12px;color:var(--faint);margin-bottom:10px}
  .legend{display:flex;flex-wrap:wrap;gap:12px;margin:2px 0 8px;font-size:12px;color:var(--muted)}
  .legend span{display:inline-flex;align-items:center;gap:6px}
  .legend i{width:12px;height:3px;border-radius:2px;display:inline-block}
  .legend b{color:var(--ink);font-variant-numeric:tabular-nums}
  canvas{width:100%;display:block}
  .kv{display:flex;flex-wrap:wrap;gap:8px 22px;font-size:12px;color:var(--muted);margin-top:10px}
  .kv b{color:var(--ink);font-variant-numeric:tabular-nums}
  .chartwrap{position:relative}
  .tip{position:absolute;pointer-events:none;background:#04101f;border:1px solid var(--line);
    border-radius:8px;padding:6px 9px;font-size:12px;color:var(--ink);opacity:0;transition:opacity .08s;
    white-space:nowrap;font-variant-numeric:tabular-nums;z-index:5;box-shadow:0 6px 18px rgba(0,0,0,.4)}
  .splitbar{display:flex;height:26px;border-radius:8px;overflow:hidden;border:1px solid var(--line)}
  .splitbar div{display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:700;color:#04101f}
  @media(max-width:880px){.grid{grid-template-columns:1fr}}
  .foot{margin-top:14px;color:var(--faint);font-size:12px}
</style></head>
<body><div class="wrap">
  <header class="top">
    <span class="title">Vimaan NLU <small>· training monitor</small></span>
    <span id="st" class="pill wait">WAITING</span>
    <span class="badge" id="b-run">run —</span>
    <span class="badge" id="b-dev">device —</span>
    <span class="badge" id="b-ep">epoch —</span>
    <span class="badge" id="b-step">step —</span>
    <span class="badge" id="b-best">best val —</span>
    <span class="badge" id="b-el">elapsed —</span>
    <span class="spacer"></span>
    <span class="badge" id="b-upd">·</span>
  </header>

  <div class="grid">
    <div class="card">
      <h2>Dataset Split</h2>
      <div class="sub" id="split-sub">70 / 15 / 15 — train / val / held-out test</div>
      <div class="splitbar" id="splitbar"></div>
      <div class="kv" id="split-kv"></div>
    </div>

    <div class="card">
      <h2>Intent Distribution <span style="text-transform:none;color:var(--faint);font-weight:500">(train split)</span></h2>
      <div class="sub" id="intent-sub">—</div>
      <div class="chartwrap"><canvas id="intents"></canvas></div>
    </div>

    <div class="card">
      <h2>Loss</h2>
      <div class="legend" id="loss-leg">
        <span><i style="background:var(--blue)"></i>train (step) <b id="lv-step">—</b></span>
        <span><i style="background:var(--blue);opacity:.55"></i>train (epoch) <b id="lv-tr">—</b></span>
        <span><i style="background:var(--orange)"></i>val (epoch) <b id="lv-val">—</b></span>
      </div>
      <div class="chartwrap"><canvas id="loss"></canvas><div class="tip" id="loss-tip"></div></div>
    </div>

    <div class="card">
      <h2>Validation Metrics</h2>
      <div class="legend" id="met-leg">
        <span><i style="background:var(--blue)"></i>intent accuracy <b id="mv-acc">—</b></span>
        <span><i style="background:var(--green)"></i>slot F1 (macro) <b id="mv-f1">—</b></span>
      </div>
      <div class="chartwrap"><canvas id="metrics"></canvas><div class="tip" id="met-tip"></div></div>
    </div>
  </div>
  <div class="foot" id="foot">Polling every 1.5s · charts are colour-blind-safe (Okabe–Ito order)</div>
</div>
<script>
const $=id=>document.getElementById(id);
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const COL={blue:css('--blue'),orange:css('--orange'),green:css('--green'),purple:css('--purple'),
  line:css('--line'),ink:css('--ink'),muted:css('--muted'),faint:css('--faint')};
const fmt=(x,d=3)=>x==null||isNaN(x)?'—':Number(x).toFixed(d);
const pct=x=>x==null||isNaN(x)?'—':(x*100).toFixed(2)+'%';
function hms(s){s=Math.max(0,Math.floor(s||0));const h=(s/3600)|0,m=((s%3600)/60)|0,ss=s%60;
  return (h?h+'h ':'')+(h||m?m+'m ':'')+ss+'s';}

function setupCanvas(cv,h){
  const dpr=window.devicePixelRatio||1, w=cv.clientWidth||cv.parentNode.clientWidth;
  cv.width=w*dpr; cv.height=h*dpr; cv.style.height=h+'px';
  const ctx=cv.getContext('2d'); ctx.setTransform(dpr,0,0,dpr,0,0);
  ctx.clearRect(0,0,w,h); return {ctx,w,h};
}
function niceMax(v){if(v<=0)return 1;const p=Math.pow(10,Math.floor(Math.log10(v)));const n=v/p;
  const s=n<=1?1:n<=2?2:n<=5?5:10;return s*p;}
function quantile(arr,q){if(!arr.length)return 0;const a=arr.slice().sort((x,y)=>x-y);
  const i=(a.length-1)*q,lo=Math.floor(i),hi=Math.ceil(i);return a[lo]+(a[hi]-a[lo])*(i-lo);}

// Generic line chart. series:[{name,color,pts:[[x,y]],markers,alpha,width}]
function lineChart(cv,series,o){
  const {ctx,w,h}=setupCanvas(cv,o.height||230);
  const L=44,R=12,T=12,B=26, pw=w-L-R, ph=h-T-B;
  let xmin=o.xmin,xmax=o.xmax,ymin=o.ymin,ymax=o.ymax;
  if(xmin==null){xmin=Infinity;xmax=-Infinity;series.forEach(s=>s.pts.forEach(p=>{xmin=Math.min(xmin,p[0]);xmax=Math.max(xmax,p[0]);}));}
  if(!isFinite(xmin)){xmin=0;xmax=1;} if(xmax===xmin)xmax=xmin+1;
  const ys=[]; series.forEach(s=>s.pts.forEach(p=>{if(isFinite(p[1]))ys.push(p[1]);}));
  if(o.yadapt==='loss'&&ys.length){
    // Ignore the one-time warmup spike: fit to the 97th percentile of losses so
    // the meaningful low-loss band fills the chart (the spike simply clips).
    if(ymin==null)ymin=0;
    if(ymax==null)ymax=quantile(ys,0.97)*1.25;
    if(!isFinite(ymax)||ymax<=0)ymax=Math.max(...ys)||1;
  }else if(o.yadapt==='zoom'&&ys.length){
    // Zoom to the data (with padding) so ~99% lines aren't flat at the top.
    let lo=Math.min(...ys),hi=Math.max(...ys),pad=Math.max((hi-lo)*0.6,0.02);
    if(ymin==null)ymin=lo-pad; if(ymax==null)ymax=hi+pad;
    if(o.yclamp){ymin=Math.max(o.yclamp[0],ymin);ymax=Math.min(o.yclamp[1],ymax);}
    const span=o.yminspan||0.08;
    if(ymax-ymin<span){const c=(ymax+ymin)/2;ymin=c-span/2;ymax=c+span/2;
      if(o.yclamp){ymin=Math.max(o.yclamp[0],ymin);ymax=Math.min(o.yclamp[1],ymax);}}
  }else{
    if(ymin==null)ymin=0;
    if(ymax==null){ymax=-Infinity;ys.forEach(v=>ymax=Math.max(ymax,v));ymax=niceMax(ymax*1.08);}
  }
  if(!isFinite(ymax)||ymax<=ymin)ymax=ymin+1;
  const X=x=>L+(x-xmin)/(xmax-xmin)*pw, Y=y=>T+ph-(y-ymin)/(ymax-ymin)*ph;
  // grid + y ticks
  ctx.font='11px -apple-system,system-ui,sans-serif'; ctx.textBaseline='middle';
  const ny=o.yticks||4, span=ymax-ymin;
  const ydec=span<0.15?(o.ypct?1:3):span<3?2:0;
  for(let i=0;i<=ny;i++){const yv=ymin+span*i/ny, y=Y(yv);
    ctx.strokeStyle=COL.line; ctx.globalAlpha=i===0?.9:.35; ctx.beginPath();ctx.moveTo(L,y);ctx.lineTo(w-R,y);ctx.stroke();
    ctx.globalAlpha=1; ctx.fillStyle=COL.faint; ctx.textAlign='right';
    ctx.fillText(o.ypct?(yv*100).toFixed(span<0.15?1:0)+'%':(''+ (+yv.toFixed(ydec))),L-6,y);
  }
  // x ticks — integer stepping for epoch axes (no duplicates), else even spacing
  ctx.textAlign='center'; ctx.textBaseline='top'; ctx.fillStyle=COL.faint;
  let xt=[];
  if(o.xint){const st=Math.max(1,Math.ceil((xmax-xmin)/8));
    for(let v=Math.ceil(xmin);v<=Math.floor(xmax)+1e-6;v+=st)xt.push(v);}
  else{const nx=o.xticks||5;for(let i=0;i<=nx;i++)xt.push(xmin+(xmax-xmin)*i/nx);}
  xt.forEach(xv=>ctx.fillText(o.xfmt?o.xfmt(xv):(''+Math.round(xv)),X(xv),h-B+7));
  // clip to plot
  ctx.save();ctx.beginPath();ctx.rect(L,T,pw,ph);ctx.clip();
  series.forEach(s=>{
    if(!s.pts.length)return;
    ctx.globalAlpha=s.alpha==null?1:s.alpha; ctx.strokeStyle=s.color; ctx.lineWidth=s.width||2;
    ctx.lineJoin='round';ctx.beginPath();
    s.pts.forEach((p,i)=>{const x=X(p[0]),y=Y(p[1]); i?ctx.lineTo(x,y):ctx.moveTo(x,y);}); ctx.stroke();
    if(s.markers){ctx.globalAlpha=1;ctx.fillStyle=s.color;
      s.pts.forEach(p=>{ctx.beginPath();ctx.arc(X(p[0]),Y(p[1]),3.2,0,7);ctx.fill();});}
  });
  ctx.restore();ctx.globalAlpha=1;
  cv._tf={X,Y,xmin,xmax,ymin,ymax,L,R,T,B,pw,ph};
}

function hbars(cv,items,o){
  const H=Math.max(60,items.length*22+8);
  const {ctx,w,h}=setupCanvas(cv,H);
  const maxv=Math.max(1,...items.map(d=>d.v));
  const labelW=Math.min(150,Math.max(60,...items.map(d=>d.k.length*6.5)));
  const L=labelW+8,R=44,bh=14,gap=8;
  ctx.font='12px -apple-system,system-ui,sans-serif';ctx.textBaseline='middle';
  items.forEach((d,i)=>{const y=8+i*(bh+gap)+bh/2, bw=(w-L-R)*d.v/maxv;
    ctx.fillStyle=COL.faint;ctx.textAlign='right';ctx.fillText(d.k,L-8,y);
    ctx.fillStyle=COL.line;ctx.globalAlpha=.5;roundRect(ctx,L,y-bh/2,w-L-R,bh,4);ctx.fill();ctx.globalAlpha=1;
    ctx.fillStyle=o.color||COL.blue;roundRect(ctx,L,y-bh/2,Math.max(3,bw),bh,4);ctx.fill();
    ctx.fillStyle=COL.muted;ctx.textAlign='left';ctx.fillText(''+d.v,L+bw+6,y);
  });
}
function roundRect(ctx,x,y,w,h,r){r=Math.min(r,h/2,w/2);ctx.beginPath();
  ctx.moveTo(x+r,y);ctx.arcTo(x+w,y,x+w,y+h,r);ctx.arcTo(x+w,y+h,x,y+h,r);
  ctx.arcTo(x,y+h,x,y,r);ctx.arcTo(x,y,x+w,y,r);ctx.closePath();}

let LAST=null;
function render(d){
  LAST=d;
  const meta=d.meta, done=d.done, steps=d.steps||[], epochs=d.epochs||[];
  const st=$('st');
  if(d.waiting||!meta){st.className='pill wait';st.textContent='WAITING FOR RUN';
    $('b-run').textContent='run —'; return;}
  const running=!done;
  st.className='pill '+(running?'run':'done'); st.textContent=running?'● TRAINING':'✓ DONE';
  const lastStep=steps.length?steps[steps.length-1]:null;
  const el=(done?done.elapsed:(lastStep?lastStep.elapsed:0));
  const curEp=epochs.length?epochs[epochs.length-1].epoch:(lastStep?lastStep.epoch:0);
  const best=epochs.reduce((m,e)=>Math.min(m,e.val_loss),Infinity);
  $('b-run').innerHTML='run <b>'+meta.version+'</b> · '+meta.dataset;
  $('b-dev').innerHTML='device <b>'+meta.device+'</b>';
  $('b-ep').innerHTML='epoch <b>'+curEp+'</b> / '+meta.epochs;
  $('b-step').innerHTML='step <b>'+(lastStep?lastStep.step:0).toLocaleString()+'</b> / '+(meta.steps_per_epoch*meta.epochs).toLocaleString();
  $('b-best').innerHTML='best val <b>'+fmt(isFinite(best)?best:null)+'</b>';
  $('b-el').innerHTML='elapsed <b>'+hms(el)+'</b>';
  $('b-upd').textContent='updated '+new Date().toLocaleTimeString();

  // split
  const sp=meta.split, tot=sp.train+sp.val+sp.test;
  $('split-sub').textContent=meta.num_intents+' intents · '+meta.num_slots+' slot labels · '+tot.toLocaleString()+' examples';
  const segs=[['train',sp.train,COL.blue],['val',sp.val,COL.orange],['test',sp.test,COL.purple]];
  $('splitbar').innerHTML=segs.map(s=>'<div style="width:'+(100*s[1]/tot)+'%;background:'+s[2]+'">'+
     (100*s[1]/tot>8?(100*s[1]/tot).toFixed(0)+'%':'')+'</div>').join('');
  $('split-kv').innerHTML=segs.map(s=>'<span style="color:'+s[2]+'">■</span>&nbsp;'+s[0]+
     ' <b>'+s[1].toLocaleString()+'</b>').join('');

  // intents
  const items=Object.entries(meta.intent_dist||{}).slice(0,15).map(([k,v])=>({k,v}));
  $('intent-sub').textContent='top '+items.length+' of '+meta.num_intents+' intents';
  hbars($('intents'),items,{color:COL.blue});

  // loss
  const spe=meta.steps_per_epoch||1;
  const stepPts=steps.map(s=>[s.step,s.loss]);
  const trPts=epochs.map(e=>[e.epoch*spe,e.train_loss]);
  const valPts=epochs.map(e=>[e.epoch*spe,e.val_loss]);
  lineChart($('loss'),[
    {name:'step',color:COL.blue,pts:stepPts,alpha:.5,width:1.4},
    {name:'train',color:COL.blue,pts:trPts,markers:true,alpha:.85,width:2},
    {name:'val',color:COL.orange,pts:valPts,markers:true,width:2},
  ],{height:236,yadapt:'loss',xfmt:v=>Math.round(v).toLocaleString()});
  $('lv-step').textContent=lastStep?fmt(lastStep.loss):'—';
  $('lv-tr').textContent=epochs.length?fmt(epochs[epochs.length-1].train_loss):'—';
  $('lv-val').textContent=epochs.length?fmt(epochs[epochs.length-1].val_loss):'—';

  // metrics
  const accPts=epochs.map(e=>[e.epoch,e.val_intent_acc]);
  const f1Pts=epochs.map(e=>[e.epoch,e.val_slot_f1]);
  lineChart($('metrics'),[
    {name:'acc',color:COL.blue,pts:accPts,markers:true,width:2},
    {name:'f1',color:COL.green,pts:f1Pts,markers:true,width:2},
  ],{height:236,ypct:true,yadapt:'zoom',yclamp:[0,1],yminspan:0.06,
     xmin:1,xmax:Math.max(2,epochs.length),xint:true});
  $('mv-acc').textContent=epochs.length?pct(epochs[epochs.length-1].val_intent_acc):'—';
  $('mv-f1').textContent=epochs.length?fmt(epochs[epochs.length-1].val_slot_f1):'—';
}

// hover tooltips
function bindTip(cv,tip,pick){
  cv.addEventListener('mousemove',ev=>{try{
    const tf=cv._tf; if(!tf||!LAST){tip.style.opacity=0;return;}
    const r=cv.getBoundingClientRect(), mx=ev.clientX-r.left;
    if(mx<tf.L||mx>tf.L+tf.pw){tip.style.opacity=0;return;}
    const xv=tf.xmin+(mx-tf.L)/tf.pw*(tf.xmax-tf.xmin);
    const html=pick(xv,tf); if(!html){tip.style.opacity=0;return;}
    tip.innerHTML=html; tip.style.opacity=1;
    let lx=mx+12; if(lx>cv.clientWidth-tip.offsetWidth-6)lx=mx-tip.offsetWidth-12;
    tip.style.left=lx+'px'; tip.style.top='10px';
  }catch(e){tip.style.opacity=0;}});
  cv.addEventListener('mouseleave',()=>tip.style.opacity=0);
}
bindTip($('loss'),$('loss-tip'),(xv)=>{
  const s=LAST.steps||[]; if(!s.length)return '';
  let b=s[0],bd=1e18; s.forEach(p=>{const dd=Math.abs(p.step-xv);if(dd<bd){bd=dd;b=p;}});
  const ep=(LAST.epochs||[]).find(e=>e.epoch===b.epoch);
  return '<b>step '+b.step.toLocaleString()+'</b> · epoch '+b.epoch+'<br>train (step): '+fmt(b.loss)+
    (ep?'<br>val (epoch '+ep.epoch+'): '+fmt(ep.val_loss):'');
});
bindTip($('metrics'),$('met-tip'),(xv)=>{
  const e=(LAST.epochs||[]); if(!e.length)return '';
  const ep=e.reduce((a,c)=>Math.abs(c.epoch-xv)<Math.abs(a.epoch-xv)?c:a,e[0]);
  return '<b>epoch '+ep.epoch+'</b><br>intent acc: '+pct(ep.val_intent_acc)+'<br>slot F1: '+fmt(ep.val_slot_f1);
});

async function tick(){
  try{const r=await fetch('/api/metrics',{cache:'no-store'});render(await r.json());}
  catch(e){$('st').textContent='monitor offline';}
}
tick(); setInterval(tick,1500);
window.addEventListener('resize',()=>LAST&&render(LAST));
</script>
</body></html>
"""


def main():
    ap = argparse.ArgumentParser(description="Live training monitor for the Vimaan NLU trainer.")
    ap.add_argument("--port", type=int, default=8788)
    ap.add_argument("--run", default=None, help="pin a run version, e.g. v11 (default: newest)")
    ap.add_argument("--runs-dir", default=RUNS_DIR)
    args = ap.parse_args()

    Handler.runs_dir = args.runs_dir
    Handler.pinned_run = args.run
    srv = DualStackServer(("::", args.port), Handler)
    where = args.run or (os.path.basename(_newest_run(args.runs_dir) or "") or "waiting for a run")
    print(f"Training monitor → http://localhost:{args.port}  or  http://127.0.0.1:{args.port}")
    print(f"  (run: {where})")
    print("Open that URL in your browser. Ctrl-C to stop.")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nmonitor stopped")


if __name__ == "__main__":
    main()
