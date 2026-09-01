"""
06_build_dashboard.py -- Build the single-file interactive dashboard.

All data is inlined into the HTML, so docs/index.html is fully self-contained:
drop it on GitHub Pages, no build step, no fetch, no CORS.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"
DOCS = ROOT / "docs"
DOCS.mkdir(exist_ok=True)


def clean(records):
    out = []
    for r in records:
        out.append({k: (None if (isinstance(v, float) and not np.isfinite(v)) else
                        (float(v) if isinstance(v, (np.floating, float)) else
                         int(v) if isinstance(v, (np.integer,)) else
                         bool(v) if isinstance(v, (np.bool_, bool)) else v))
                    for k, v in r.items()})
    return out


def build_payload():
    resid = pd.read_csv(PROC / "species_residuals.csv")
    cols = ["SpeciesLatinName", "SpeciesCommonName", "Order", "adult_mass_g",
            "max_lifespan_y", "span_y", "n_samples", "n_tissues", "arocm",
            "log_mass", "log_span", "log_arocm", "implied_cor",
            "longevity_quotient", "resid_M1", "resid_M2", "resid_M3",
            "resid_student", "q_outlier", "signif", "tissues"]
    species = clean(resid[[c for c in cols if c in resid.columns]]
                    .to_dict("records"))

    pub = pd.read_csv(PROC / "clock_predictions_published.csv")
    scr = pd.read_csv(PROC / "clock_predictions_scratch.csv")
    clocks = clean(pub[["Basename", "Age", "DNAmAge_clock1", "DNAmAge_clock2",
                        "DNAmAge_clock3", "Female"]]
                   .merge(scr[["Basename", "PredAge_scratch"]], on="Basename")
                   .to_dict("records"))

    states = pd.read_csv(PROC / "state_lifespan_association.csv")
    imp = pd.read_csv(PROC / "state_rf_importance.csv")[["state", "importance",
                                                         "importance_sd"]]
    var = pd.read_csv(PROC / "state_variation.csv")[["state", "cv_across_species",
                                                     "mean_contrast"]]
    coef = pd.read_csv(PROC / "state_elasticnet_coefficients.csv")[["state", "coef"]]
    states = (states.merge(imp, on="state").merge(var, on="state")
              .merge(coef, on="state"))

    enrich = clean(pd.read_csv(PROC / "clock_cpg_enrichment.csv").to_dict("records"))
    ann = pd.read_csv(PROC / "clock_cpg_annotation.csv")
    genes = clean(ann.SYMBOL.value_counts().head(25).reset_index()
                  .rename(columns={"index": "gene", "SYMBOL": "gene",
                                   "count": "n"}).to_dict("records"))

    return {
        "species": species,
        "clocks": clocks,
        "states": clean(states.to_dict("records")),
        "enrichment": enrich,
        "genes": genes,
        "models": json.loads((RES / "aging_rate_models.json").read_text()),
        "clock_stats": json.loads((RES / "clock_validation.json").read_text()),
        "interp": json.loads((RES / "interpretability.json").read_text()),
        "manifest": json.loads((PROC / "ingest_manifest.json").read_text()),
    }


HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Mammalian Methylation Clocks &middot; What the residuals actually say</title>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>
<style>
:root{
  --bg:#0d0e1a; --panel:#151729; --panel2:#1b1e35; --line:#252842;
  --fg:#e6e6f0; --muted:#8b8ba7;
  --indigo:#6366f1; --violet:#a78bfa; --cyan:#22d3ee;
  --amber:#fbbf24; --rose:#fb7185; --emerald:#34d399;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
  font-family:'Space Grotesk',system-ui,sans-serif;line-height:1.55;
  -webkit-font-smoothing:antialiased}
.wrap{max-width:1180px;margin:0 auto;padding:36px 22px 80px}
header{border-bottom:1px solid var(--line);padding-bottom:26px;margin-bottom:26px}
h1{font-size:31px;margin:0 0 8px;letter-spacing:-.02em;font-weight:700}
.sub{color:var(--muted);font-size:15px;max-width:76ch;margin:0}
.mono{font-family:'JetBrains Mono',monospace}
.tabs{display:flex;gap:4px;flex-wrap:wrap;margin:26px 0 22px;
  border-bottom:1px solid var(--line)}
.tab{padding:9px 15px;cursor:pointer;color:var(--muted);font-size:14px;
  border-bottom:2px solid transparent;transition:.15s;user-select:none;font-weight:500}
.tab:hover{color:var(--fg)}
.tab.active{color:var(--violet);border-bottom-color:var(--violet)}
.panel{display:none;animation:fade .25s ease}
.panel.active{display:block}
@keyframes fade{from{opacity:0;transform:translateY(4px)}to{opacity:1;transform:none}}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(178px,1fr));
  gap:13px;margin:22px 0}
.card{background:var(--panel);border:1px solid var(--line);border-radius:11px;
  padding:16px 17px}
.card .v{font-size:26px;font-weight:700;font-family:'JetBrains Mono',monospace;
  letter-spacing:-.02em}
.card .k{color:var(--muted);font-size:11.5px;text-transform:uppercase;
  letter-spacing:.07em;margin-top:5px}
.card .n{color:var(--muted);font-size:11.5px;margin-top:7px;line-height:1.4}
.chart{background:var(--panel);border:1px solid var(--line);border-radius:12px;
  padding:18px;margin:16px 0}
.chart h3{margin:0 0 3px;font-size:16px}
.chart .cap{color:var(--muted);font-size:12.5px;margin:0 0 14px;max-width:88ch}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media(max-width:900px){.grid2{grid-template-columns:1fr}}
.note{background:var(--panel2);border-left:3px solid var(--amber);
  border-radius:0 9px 9px 0;padding:14px 17px;margin:18px 0;font-size:14px}
.note b{color:var(--amber)}
.note.good{border-left-color:var(--emerald)} .note.good b{color:var(--emerald)}
.note.info{border-left-color:var(--indigo)} .note.info b{color:var(--violet)}
table{width:100%;border-collapse:collapse;font-size:13px}
th{text-align:left;color:var(--muted);font-weight:500;font-size:11.5px;
  text-transform:uppercase;letter-spacing:.06em;padding:9px 10px;
  border-bottom:1px solid var(--line);cursor:pointer;user-select:none}
th:hover{color:var(--fg)}
td{padding:8px 10px;border-bottom:1px solid rgba(37,40,66,.55)}
tr:hover td{background:rgba(99,102,241,.07)}
.num{font-family:'JetBrains Mono',monospace;text-align:right}
.pos{color:var(--rose)} .neg{color:var(--indigo)}
input[type=search]{background:var(--panel2);border:1px solid var(--line);
  color:var(--fg);padding:8px 12px;border-radius:8px;width:250px;
  font-family:'Space Grotesk';font-size:13px;margin-bottom:12px}
input[type=search]:focus{outline:none;border-color:var(--violet)}
.tip{position:absolute;pointer-events:none;background:#0a0b14ee;
  border:1px solid var(--line);border-radius:8px;padding:9px 11px;font-size:12px;
  opacity:0;transition:opacity .12s;z-index:50;max-width:250px;
  box-shadow:0 8px 26px #0008}
.tip .t{font-weight:700;margin-bottom:4px}
.tip .r{color:var(--muted);font-size:11.5px;font-family:'JetBrains Mono',monospace}
.axis text{fill:var(--muted);font-size:11px;font-family:'JetBrains Mono',monospace}
.axis line,.axis path{stroke:var(--line)}
.glab{fill:var(--muted);font-size:12px}
.legend{display:flex;gap:15px;flex-wrap:wrap;font-size:12px;color:var(--muted);
  margin-top:10px}
.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;
  margin-right:5px;vertical-align:middle}
.toggle{display:flex;gap:6px;margin-bottom:12px;flex-wrap:wrap}
.toggle button{background:var(--panel2);border:1px solid var(--line);
  color:var(--muted);padding:6px 12px;border-radius:7px;cursor:pointer;
  font-family:'Space Grotesk';font-size:12.5px;transition:.15s}
.toggle button:hover{color:var(--fg)}
.toggle button.on{background:var(--indigo);color:#fff;border-color:var(--indigo)}
footer{margin-top:46px;padding-top:22px;border-top:1px solid var(--line);
  color:var(--muted);font-size:12.5px}
a{color:var(--violet)}
</style>
</head>
<body>
<div class="wrap">
<header>
  <h1>What the residuals actually say</h1>
  <p class="sub">A pan-mammalian methylation clock, the body-size question, and a
  measurement artifact that eats most of the answer. Built on real data from the
  Mammalian Methylation Consortium &mdash; 126 species, 11,112 arrays,
  54 chromatin states.</p>
</header>

<div class="tabs" id="tabs"></div>

<!-- ============ OVERVIEW ============ -->
<div class="panel" id="p-overview">
  <div class="cards" id="ov-cards"></div>
  <div class="note info"><b>The question.</b> Methylation clocks predict age
  across mammals with startling accuracy. So which species age faster or slower
  than their body size predicts? Naked mole rats and bowhead whales are the
  usual suspects. The answer turned out to depend almost entirely on how the
  aging rate is measured.</div>
  <div class="note"><b>The catch.</b> The published aging-rate measure (AROCM) is
  the slope of <i>z-scored</i> methylation on age. Standardising inside each
  species makes that slope mechanically proportional to 1/SD(sampled ages).
  Long-lived species get sampled across wider spans in years, so their rate is
  pushed down for reasons that have nothing to do with biology. Fitting it
  directly, the coefficient on sampled span is
  <span class="mono" id="ov-span"></span> &mdash; statistically indistinguishable
  from the &minus;1 a pure artifact predicts.</div>
  <div class="note good"><b>What survives.</b> The clocks themselves are real and
  excellent: the published universal clock hits r&nbsp;=&nbsp;0.948 on held-out
  dolphin blood, and a clock trained here from scratch reaches r&nbsp;=&nbsp;0.915
  while independently rediscovering CpGs the published clocks use. The CpGs sit
  in developmental transcription factors. What does <i>not</i> survive is the
  cross-species story built on top of the rate measure.</div>
  <div class="chart">
    <h3>The argument in one chart</h3>
    <p class="cap">Left: epigenetic aging rate against body mass &mdash; a clean,
    highly significant allometry. Right: the same rate against the span of ages
    the species was sampled over, with the dotted line showing the slope of
    &minus;1 expected if the relationship were purely an artifact of the
    measure. Hover any point.</p>
    <div class="grid2">
      <div id="c-mass"></div><div id="c-span"></div>
    </div>
  </div>
</div>

<!-- ============ CLOCKS ============ -->
<div class="panel" id="p-clocks">
  <div class="cards" id="ck-cards"></div>
  <div class="chart">
    <h3>Predicted versus chronological age</h3>
    <p class="cap">50 real bottlenose-dolphin blood samples, ages 0.57 to 57.9
    years, on the 37,554-CpG mammalian array. The from-scratch model is scored
    leave-one-out, so every point is a genuine held-out prediction.</p>
    <div class="toggle" id="ck-toggle"></div>
    <div id="c-clock"></div>
  </div>
  <div class="note good"><b>Independent rediscovery.</b> The from-scratch clock
  selected <span class="mono" id="ck-n"></span> CpGs. <span class="mono"
  id="ck-ov"></span> of them also appear in the published universal clocks,
  against <span class="mono" id="ck-exp"></span> expected by chance
  (hypergeometric p&nbsp;=&nbsp;<span class="mono" id="ck-p"></span>). Two
  independent fits on the same biology converge on the same sites.</div>
</div>

<!-- ============ ARTIFACT ============ -->
<div class="panel" id="p-artifact">
  <div class="chart">
    <h3>Four models, increasing scepticism</h3>
    <p class="cap">Each row adds one control. The mass coefficient is the answer
    to "does body size predict epigenetic aging rate". Watch it collapse.</p>
    <div id="c-models"></div>
  </div>
  <div class="chart">
    <h3>Rate versus the sampled age span</h3>
    <p class="cap">Colour by taxonomic order. The relationship holds within every
    order, which is what a mechanical artifact looks like &mdash; biology rarely
    lines up this tidily.</p>
    <div id="c-spanbig"></div>
    <div class="legend" id="l-order"></div>
  </div>
</div>

<!-- ============ RESIDUALS ============ -->
<div class="panel" id="p-residuals">
  <div class="note"><b>Read this column carefully.</b> After the span term is
  removed, the residual no longer measures how <i>fast</i> a species ages. It
  measures how tightly its methylation tracks age &mdash; clock fidelity, not
  clock speed. Negative means the epigenome is a looser age predictor than body
  size would suggest.</div>
  <div class="chart">
    <h3>Every species, naive versus corrected</h3>
    <p class="cap">Horizontal axis: residual from the naive mass-only model.
    Vertical: after controlling for sampled span and taxonomic order. Points far
    from the diagonal are species whose reputation depends on the artifact.</p>
    <div id="c-resid"></div>
  </div>
  <div class="chart">
    <h3>Ranked table</h3>
    <p class="cap">Click a header to sort. Only species marked significant
    survive Benjamini-Hochberg correction across all 126 tests.</p>
    <input type="search" id="sp-search" placeholder="Filter species or order...">
    <div style="max-height:520px;overflow:auto"><table id="t-species"></table></div>
  </div>
</div>

<!-- ============ INTERPRETABILITY ============ -->
<div class="panel" id="p-interp">
  <div class="note info"><b>A cleaner signal.</b> Dividing each chromatin state's
  rate by the species' own genome-wide mean cancels the 1/SD(age) term exactly.
  What is left is the <i>shape</i> of the aging signature &mdash; which parts of
  the genome drift fastest, relative to everything else in that animal.</div>
  <div class="chart">
    <h3>Which chromatin states carry cross-species differences</h3>
    <p class="cap">Bar length is random-forest permutation importance for
    predicting maximum lifespan. Colour marks the Spearman correlation with
    lifespan. Hover for the numbers.</p>
    <div id="c-states"></div>
  </div>
  <div class="note"><b>And why it does not hold up.</b> Six of 54 states
  correlate with lifespan at q&nbsp;&lt;&nbsp;0.05. But when whole taxonomic
  orders are held out &mdash; so the model must generalise to a lineage it has
  never seen &mdash; out-of-sample r is
  <span class="mono" id="in-r"></span> (permutation p&nbsp;=&nbsp;<span
  class="mono" id="in-p"></span>). The associations are phylogenetic structure,
  not a portable signature.</div>
  <div class="grid2">
    <div class="chart">
      <h3>Where the clock CpGs sit</h3>
      <p class="cap">1,303 CpGs from the three universal clocks against the full
      array background. Bars right of zero are enriched.</p>
      <div id="c-enrich"></div>
    </div>
    <div class="chart">
      <h3>Genes with the most clock CpGs</h3>
      <p class="cap">Developmental transcription factors and polycomb targets
      &mdash; the signature every methylation clock converges on.</p>
      <div id="c-genes"></div>
    </div>
  </div>
</div>

<footer>
  Data: <a href="https://github.com/shorvath/MammalianMethylationConsortium">Mammalian
  Methylation Consortium</a> public release (Lu et al., <i>Nature Aging</i> 2023;
  Zhe Fei et al., fundamental equations). Chromatin states from the Ernst lab
  annotation. Every figure on this page is computed from real consortium data;
  nothing is simulated.
</footer>
</div>

<div class="tip" id="tip"></div>
<script>
const DATA = __DATA__;
const C={indigo:'#6366f1',violet:'#a78bfa',cyan:'#22d3ee',amber:'#fbbf24',
         rose:'#fb7185',emerald:'#34d399',muted:'#8b8ba7',line:'#252842',fg:'#e6e6f0'};
const tip=d3.select('#tip');
function showTip(ev,html){tip.html(html).style('opacity',1)
  .style('left',(ev.pageX+14)+'px').style('top',(ev.pageY-10)+'px');}
function hideTip(){tip.style('opacity',0);}
const fmt=(v,d=2)=>v==null?'--':(+v).toFixed(d);

/* ---------- tabs ---------- */
const TABS=[['overview','Overview'],['clocks','The clocks work'],
  ['artifact','The artifact'],['residuals','Residuals'],
  ['interp','Interpretability']];
const drawn={};
d3.select('#tabs').selectAll('div').data(TABS).join('div')
  .attr('class',(d,i)=>'tab'+(i===0?' active':'')).text(d=>d[1])
  .on('click',(e,d)=>activate(d[0]));
function activate(id){
  d3.selectAll('.tab').classed('active',(d)=>d[0]===id);
  d3.selectAll('.panel').classed('active',false);
  d3.select('#p-'+id).classed('active',true);
  // panels are display:none on load, so D3 measures zero width -- redraw here
  if(!drawn[id]){ RENDER[id](); drawn[id]=true; }
}

/* ---------- helpers ---------- */
function frame(sel,h,m){
  const el=document.querySelector(sel);
  const w=el.clientWidth||760;
  d3.select(sel).selectAll('svg').remove();
  const svg=d3.select(sel).append('svg').attr('width',w).attr('height',h);
  const g=svg.append('g').attr('transform',`translate(${m.l},${m.t})`);
  return {svg,g,w:w-m.l-m.r,h:h-m.t-m.b};
}
function axes(g,x,y,w,h,xl,yl){
  g.append('g').attr('class','axis').attr('transform',`translate(0,${h})`)
    .call(d3.axisBottom(x).ticks(6).tickSize(-h).tickPadding(8))
    .call(s=>s.selectAll('.tick line').attr('stroke',C.line).attr('stroke-opacity',.5))
    .call(s=>s.select('.domain').remove());
  g.append('g').attr('class','axis')
    .call(d3.axisLeft(y).ticks(6).tickSize(-w).tickPadding(8))
    .call(s=>s.selectAll('.tick line').attr('stroke',C.line).attr('stroke-opacity',.5))
    .call(s=>s.select('.domain').remove());
  g.append('text').attr('class','glab').attr('x',w/2).attr('y',h+40)
    .attr('text-anchor','middle').text(xl);
  g.append('text').attr('class','glab').attr('transform','rotate(-90)')
    .attr('x',-h/2).attr('y',-46).attr('text-anchor','middle').text(yl);
}
function fitLine(g,x,y,data,ax,ay,col,dash){
  const n=data.length, sx=d3.sum(data,d=>d[ax]), sy=d3.sum(data,d=>d[ay]);
  const sxx=d3.sum(data,d=>d[ax]*d[ax]), sxy=d3.sum(data,d=>d[ax]*d[ay]);
  const b=(n*sxy-sx*sy)/(n*sxx-sx*sx), a=(sy-b*sx)/n;
  const xs=d3.extent(data,d=>d[ax]);
  g.append('line').attr('x1',x(xs[0])).attr('y1',y(a+b*xs[0]))
    .attr('x2',x(xs[1])).attr('y2',y(a+b*xs[1]))
    .attr('stroke',col).attr('stroke-width',2)
    .attr('stroke-dasharray',dash||null);
  return {a,b};
}
const ORDCOL=d3.scaleOrdinal().range([C.indigo,C.violet,C.cyan,C.amber,
  C.emerald,C.rose,'#f472b6','#38bdf8','#a3e635']);

/* ---------- scatter of species ---------- */
function speciesScatter(sel,ax,ay,xl,yl,h,opts){
  opts=opts||{};
  const d=DATA.species.filter(s=>s[ax]!=null&&s[ay]!=null);
  const F=frame(sel,h,{l:64,r:16,t:14,b:52});
  const x=d3.scaleLinear().domain(d3.extent(d,s=>s[ax])).nice().range([0,F.w]);
  const y=d3.scaleLinear().domain(d3.extent(d,s=>s[ay])).nice().range([F.h,0]);
  axes(F.g,x,y,F.w,F.h,xl,yl);
  if(opts.artifactLine){
    const c=d3.mean(d,s=>s.log_arocm+s.log_span);
    const xs=x.domain();
    F.g.append('line').attr('x1',x(xs[0])).attr('y1',y(c-xs[0]))
      .attr('x2',x(xs[1])).attr('y2',y(c-xs[1]))
      .attr('stroke',C.emerald).attr('stroke-width',2)
      .attr('stroke-dasharray','5,4');
  }
  fitLine(F.g,x,y,d,ax,ay,C.amber);
  F.g.selectAll('circle').data(d).join('circle')
    .attr('cx',s=>x(s[ax])).attr('cy',s=>y(s[ay]))
    .attr('r',opts.r||4.2)
    .attr('fill',s=>opts.byOrder?ORDCOL(s.Order):(opts.col||C.indigo))
    .attr('fill-opacity',.8).attr('stroke','#0d0e1a').attr('stroke-width',.6)
    .on('mousemove',(e,s)=>showTip(e,
      `<div class="t">${s.SpeciesCommonName}</div>
       <div class="r">${s.Order} &middot; ${s.n_samples} arrays<br>
       mass ${d3.format(',')(Math.round(s.adult_mass_g))} g<br>
       lifespan ${fmt(s.max_lifespan_y,1)} y<br>
       sampled span ${fmt(s.span_y,1)} y</div>`))
    .on('mouseleave',hideTip);
  return F;
}

/* ---------- renderers ---------- */
const RENDER={};

RENDER.overview=function(){
  const m=DATA.models, ck=DATA.clock_stats, mf=DATA.manifest;
  const cards=[
    ['126','species','15 taxonomic orders'],
    [d3.format(',')(mf.dolphin_betas.cpgs),'CpGs per array','mammalian methylation chip'],
    ['11,112','arrays','behind the species-level rates'],
    ['0.948','published clock r','on held-out dolphin blood'],
    ['89%','of the allometry','vanishes once sampling is controlled'],
    ['1 / 126','real outliers','after multiple-testing correction']];
  d3.select('#ov-cards').selectAll('.card').data(cards).join('div')
    .attr('class','card').html(d=>
      `<div class="v">${d[0]}</div><div class="k">${d[1]}</div>
       <div class="n">${d[2]}</div>`);
  const sp=m.models.M2;
  d3.select('#ov-span').text(
    `${fmt(sp.params.log_span,3)} ± ${fmt(sp.se.log_span,3)}`);
  speciesScatter('#c-mass','log_mass','log_arocm',
    'log body mass (g)','log aging rate |AROCM|',330,{col:C.indigo});
  speciesScatter('#c-span','log_span','log_arocm',
    'log sampled age span (y)','log aging rate |AROCM|',330,
    {col:C.rose,artifactLine:true});
};

RENDER.clocks=function(){
  const s=DATA.clock_stats;
  const cards=[
    ['r = '+fmt(s.published_clocks.clock2.pearson_r,3),'universal clock 2',
     fmt(s.published_clocks.clock2.median_AE,2)+' y median error'],
    ['r = '+fmt(s.from_scratch_clock.loo_pearson_r,3),'trained here (LOO-CV)',
     fmt(s.from_scratch_clock.loo_median_AE_years,2)+' y median error'],
    [s.from_scratch_clock.n_cpgs_selected,'CpGs chosen','out of 37,554'],
    ['p = '+fmt(s.from_scratch_clock.permutation.empirical_p,3),'permutation test',
     '100 label shuffles']];
  d3.select('#ck-cards').selectAll('.card').data(cards).join('div')
    .attr('class','card').html(d=>
      `<div class="v">${d[0]}</div><div class="k">${d[1]}</div>
       <div class="n">${d[2]}</div>`);
  const ov=s.from_scratch_clock.overlap_with_published;
  d3.select('#ck-n').text(ov.n_selected);
  d3.select('#ck-ov').text(ov.n_overlap);
  d3.select('#ck-exp').text(fmt(ov.expected_by_chance,1));
  d3.select('#ck-p').text(d3.format('.1e')(ov.hypergeom_p));

  const SERIES=[['DNAmAge_clock2','Universal clock 2',C.violet],
                ['PredAge_scratch','Trained here (LOO)',C.cyan],
                ['DNAmAge_clock3','Universal clock 3',C.amber],
                ['DNAmAge_clock1','Universal clock 1',C.rose]];
  let active=new Set(['DNAmAge_clock2','PredAge_scratch']);
  d3.select('#ck-toggle').selectAll('button').data(SERIES).join('button')
    .attr('class',d=>active.has(d[0])?'on':'').text(d=>d[1])
    .on('click',function(e,d){
      active.has(d[0])?active.delete(d[0]):active.add(d[0]);
      d3.select(this).classed('on',active.has(d[0])); drawClock();});
  function drawClock(){
    const F=frame('#c-clock',420,{l:64,r:16,t:14,b:52});
    const vals=[].concat(...SERIES.filter(s=>active.has(s[0]))
      .map(s=>DATA.clocks.map(c=>c[s[0]])));
    const mx=Math.max(d3.max(DATA.clocks,c=>c.Age),d3.max(vals)||0)*1.06;
    const x=d3.scaleLinear().domain([0,mx]).range([0,F.w]);
    const y=d3.scaleLinear().domain([0,mx]).range([F.h,0]);
    axes(F.g,x,y,F.w,F.h,'chronological age (years)','predicted age (years)');
    F.g.append('line').attr('x1',x(0)).attr('y1',y(0)).attr('x2',x(mx))
      .attr('y2',y(mx)).attr('stroke',C.muted).attr('stroke-dasharray','4,4');
    SERIES.filter(s=>active.has(s[0])).forEach(s=>{
      F.g.selectAll('c'+s[0]).data(DATA.clocks).join('circle')
        .attr('cx',c=>x(c.Age)).attr('cy',c=>y(c[s[0]])).attr('r',5)
        .attr('fill',s[2]).attr('fill-opacity',.82)
        .attr('stroke','#0d0e1a').attr('stroke-width',.7)
        .on('mousemove',(e,c)=>showTip(e,
          `<div class="t">${s[1]}</div><div class="r">
           actual ${fmt(c.Age,1)} y<br>predicted ${fmt(c[s[0]],1)} y<br>
           error ${fmt(c[s[0]]-c.Age,1)} y</div>`))
        .on('mouseleave',hideTip);});
  }
  drawClock();
};

RENDER.artifact=function(){
  const m=DATA.models.models;
  const rows=[
    {n:'M1  mass only',v:m.M1.params.log_mass,se:m.M1.se.log_mass,
     p:m.M1.pvalues.log_mass,note:'naive'},
    {n:'M2  + sampled span',v:m.M2.params.log_mass,se:m.M2.se.log_mass,
     p:m.M2.pvalues.log_mass,note:'artifact removed'},
    {n:'M3  + Order effect',v:m.M3.params.log_mass,se:m.M3.se.log_mass,
     p:m.M3.pvalues.log_mass,note:'phylogeny controlled'},
    {n:'M5  implied correlation',v:m.M5.params.log_mass,se:m.M5.se.log_mass,
     p:m.M5.pvalues.log_mass,note:'artifact-free measure'}];
  const F=frame('#c-models',250,{l:180,r:80,t:12,b:46});
  const x=d3.scaleLinear().domain([-0.2,0.06]).range([0,F.w]);
  const y=d3.scaleBand().domain(rows.map(r=>r.n)).range([0,F.h]).padding(.42);
  F.g.append('g').attr('class','axis').attr('transform',`translate(0,${F.h})`)
    .call(d3.axisBottom(x).ticks(6)).call(s=>s.select('.domain').remove());
  F.g.append('g').attr('class','axis').call(d3.axisLeft(y).tickSize(0))
    .call(s=>s.select('.domain').remove())
    .call(s=>s.selectAll('text').attr('fill',C.fg).style('font-size','12.5px'));
  F.g.append('line').attr('x1',x(0)).attr('x2',x(0)).attr('y1',0).attr('y2',F.h)
    .attr('stroke',C.muted).attr('stroke-dasharray','4,4');
  const rg=F.g.selectAll('g.r').data(rows).join('g').attr('class','r')
    .attr('transform',r=>`translate(0,${y(r.n)+y.bandwidth()/2})`);
  rg.append('line').attr('x1',r=>x(r.v-1.96*r.se)).attr('x2',r=>x(r.v+1.96*r.se))
    .attr('stroke',r=>r.p<0.05?C.rose:C.muted).attr('stroke-width',2.4);
  rg.append('circle').attr('cx',r=>x(r.v)).attr('r',6.5)
    .attr('fill',r=>r.p<0.05?C.rose:C.indigo)
    .attr('stroke','#0d0e1a').attr('stroke-width',1);
  rg.append('text').attr('x',F.w+10).attr('dy','.35em')
    .attr('fill',C.muted).style('font-size','11.5px')
    .style('font-family','JetBrains Mono')
    .text(r=>'p='+(r.p<0.001?d3.format('.0e')(r.p):fmt(r.p,3)));
  F.g.append('text').attr('class','glab').attr('x',F.w/2).attr('y',F.h+38)
    .attr('text-anchor','middle')
    .text('coefficient on log body mass  (95% CI)');
  speciesScatter('#c-spanbig','log_span','log_arocm',
    'log sampled age span (y)','log aging rate |AROCM|',430,
    {byOrder:true,artifactLine:true,r:5});
  const orders=[...new Set(DATA.species.map(s=>s.Order))];
  d3.select('#l-order').selectAll('span').data(orders).join('span')
    .html(o=>`<i style="background:${ORDCOL(o)}"></i>${o}`);
};

RENDER.residuals=function(){
  const d=DATA.species.filter(s=>s.resid_M3!=null);
  const F=frame('#c-resid',430,{l:64,r:16,t:14,b:52});
  const ex=d3.extent(d,s=>s.resid_M1), ey=d3.extent(d,s=>s.resid_M3);
  const lo=Math.min(ex[0],ey[0])-.3, hi=Math.max(ex[1],ey[1])+.3;
  const x=d3.scaleLinear().domain([lo,hi]).range([0,F.w]);
  const y=d3.scaleLinear().domain([lo,hi]).range([F.h,0]);
  axes(F.g,x,y,F.w,F.h,'residual, naive model (mass only)',
       'residual, corrected (mass + span + order)');
  F.g.append('line').attr('x1',x(lo)).attr('y1',y(lo)).attr('x2',x(hi))
    .attr('y2',y(hi)).attr('stroke',C.line).attr('stroke-dasharray','4,4');
  [[x(lo),x(hi),y(0),y(0)],[x(0),x(0),y(lo),y(hi)]].forEach(c=>
    F.g.append('line').attr('x1',c[0]).attr('x2',c[1]).attr('y1',c[2])
      .attr('y2',c[3]).attr('stroke',C.muted).attr('stroke-opacity',.45));
  const HERO={'Heterocephalus glaber':C.amber,'Balaena mysticetus':C.cyan,
              'Homo sapiens':C.violet,'Mus musculus':C.emerald};
  F.g.selectAll('circle').data(d).join('circle')
    .attr('cx',s=>x(s.resid_M1)).attr('cy',s=>y(s.resid_M3))
    .attr('r',s=>HERO[s.SpeciesLatinName]?7.5:4)
    .attr('fill',s=>HERO[s.SpeciesLatinName]||(s.signif?C.rose:C.muted))
    .attr('fill-opacity',s=>HERO[s.SpeciesLatinName]?1:.62)
    .attr('stroke','#0d0e1a').attr('stroke-width',.8)
    .on('mousemove',(e,s)=>showTip(e,
      `<div class="t">${s.SpeciesCommonName}</div><div class="r">
       naive ${fmt(s.resid_M1)}<br>corrected ${fmt(s.resid_M3)}<br>
       q = ${fmt(s.q_outlier,3)}</div>`))
    .on('mouseleave',hideTip);
  F.g.selectAll('text.h').data(d.filter(s=>HERO[s.SpeciesLatinName]))
    .join('text').attr('class','h')
    .attr('x',s=>x(s.resid_M1)+(s.resid_M1<-1.2?-11:11))
    .attr('y',s=>y(s.resid_M3)-9)
    .attr('text-anchor',s=>s.resid_M1<-1.2?'end':'start')
    .attr('fill',s=>HERO[s.SpeciesLatinName])
    .style('font-size','12px').style('font-weight','700')
    .text(s=>s.SpeciesCommonName);

  const COLS=[['SpeciesCommonName','Species',0],['Order','Order',0],
    ['adult_mass_g','Mass (g)',0],['max_lifespan_y','Lifespan (y)',1],
    ['span_y','Sampled span (y)',1],['resid_M1','Naive resid',2],
    ['resid_M3','Corrected resid',2],['q_outlier','q',3]];
  let sortK='resid_M3', asc=true, filt='';
  function table(){
    let rows=DATA.species.filter(s=>s.resid_M3!=null);
    if(filt) rows=rows.filter(s=>
      (s.SpeciesCommonName+' '+s.Order).toLowerCase().includes(filt));
    rows.sort((a,b)=>{const va=a[sortK],vb=b[sortK];
      return (va<vb?-1:va>vb?1:0)*(asc?1:-1);});
    const t=d3.select('#t-species'); t.selectAll('*').remove();
    t.append('thead').append('tr').selectAll('th').data(COLS).join('th')
      .text(c=>c[1]+(c[0]===sortK?(asc?' ↑':' ↓'):''))
      .on('click',(e,c)=>{if(sortK===c[0])asc=!asc;else{sortK=c[0];asc=true;}table();});
    t.append('tbody').selectAll('tr').data(rows).join('tr')
      .selectAll('td').data(s=>COLS.map(c=>({s,c}))).join('td')
      .attr('class',o=>o.c[2]?'num':'')
      .classed('pos',o=>o.c[0].startsWith('resid')&&o.s[o.c[0]]>0)
      .classed('neg',o=>o.c[0].startsWith('resid')&&o.s[o.c[0]]<0)
      .html(o=>{const v=o.s[o.c[0]];
        if(o.c[2]===0&&o.c[0]==='adult_mass_g') return d3.format(',')(Math.round(v));
        if(o.c[2]===0) return v;
        if(o.c[0]==='q_outlier') return (v<0.05?'<b style="color:#fbbf24">':'')
          +fmt(v,3)+(v<0.05?' *</b>':'');
        return fmt(v,o.c[2]===1?1:2);});
  }
  d3.select('#sp-search').on('input',function(){filt=this.value.toLowerCase();table();});
  table();
};

RENDER.interp=function(){
  const st=DATA.states.slice().sort((a,b)=>b.importance-a.importance).slice(0,16);
  const F=frame('#c-states',480,{l:110,r:120,t:12,b:48});
  const x=d3.scaleLinear().domain([0,d3.max(st,s=>s.importance)*1.12]).range([0,F.w]);
  const y=d3.scaleBand().domain(st.map(s=>s.state)).range([0,F.h]).padding(.24);
  const col=d3.scaleSequential(d3.interpolateRgbBasis([C.indigo,'#3b3f6b',C.rose]))
    .domain([-0.35,0.35]);
  F.g.append('g').attr('class','axis').attr('transform',`translate(0,${F.h})`)
    .call(d3.axisBottom(x).ticks(5)).call(s=>s.select('.domain').remove());
  F.g.append('g').attr('class','axis').call(d3.axisLeft(y).tickSize(0))
    .call(s=>s.select('.domain').remove())
    .call(s=>s.selectAll('text').attr('fill',C.fg)
      .style('font-family','JetBrains Mono').style('font-size','11.5px'));
  F.g.selectAll('rect').data(st).join('rect')
    .attr('x',0).attr('y',s=>y(s.state)).attr('height',y.bandwidth())
    .attr('width',s=>x(s.importance)).attr('rx',3)
    .attr('fill',s=>col(s.spearman_rho))
    .on('mousemove',(e,s)=>showTip(e,
      `<div class="t">${s.state}</div><div class="r">
       importance ${fmt(s.importance,4)}<br>
       Spearman vs lifespan ${fmt(s.spearman_rho,3)}<br>
       q = ${fmt(s.q,4)}<br>
       cross-species CV ${fmt(s.cv_across_species,3)}</div>`))
    .on('mouseleave',hideTip);
  F.g.selectAll('text.q').data(st).join('text').attr('class','q')
    .attr('x',s=>x(s.importance)+8).attr('y',s=>y(s.state)+y.bandwidth()/2)
    .attr('dy','.35em').attr('fill',s=>s.q<0.05?C.amber:C.muted)
    .style('font-size','11px').style('font-family','JetBrains Mono')
    .text(s=>'ρ='+fmt(s.spearman_rho,2)+(s.q<0.05?' *':''));
  F.g.append('text').attr('class','glab').attr('x',F.w/2).attr('y',F.h+38)
    .attr('text-anchor','middle').text('random-forest permutation importance');
  d3.select('#in-r').text(fmt(DATA.interp.ml.elastic_net.r,3));
  d3.select('#in-p').text(fmt(DATA.interp.ml.permutation.empirical_p,2));

  const en=DATA.enrichment.slice().sort((a,b)=>a.odds_ratio-b.odds_ratio);
  const F2=frame('#c-enrich',330,{l:140,r:22,t:10,b:44});
  const lg=en.map(e=>Math.log2(e.odds_ratio));
  const x2=d3.scaleLinear().domain([d3.min(lg)*1.2,d3.max(lg)*1.25]).range([0,F2.w]);
  const y2=d3.scaleBand().domain(en.map(e=>e.category)).range([0,F2.h]).padding(.25);
  F2.g.append('g').attr('class','axis').attr('transform',`translate(0,${F2.h})`)
    .call(d3.axisBottom(x2).ticks(5)).call(s=>s.select('.domain').remove());
  F2.g.append('g').attr('class','axis').call(d3.axisLeft(y2).tickSize(0))
    .call(s=>s.select('.domain').remove())
    .call(s=>s.selectAll('text').attr('fill',C.fg).style('font-size','11px'));
  F2.g.selectAll('rect').data(en).join('rect')
    .attr('x',e=>Math.min(x2(0),x2(Math.log2(e.odds_ratio))))
    .attr('y',e=>y2(e.category)).attr('height',y2.bandwidth())
    .attr('width',e=>Math.abs(x2(Math.log2(e.odds_ratio))-x2(0))).attr('rx',3)
    .attr('fill',e=>e.odds_ratio>1?C.emerald:C.rose)
    .attr('fill-opacity',e=>e.q<0.05?.95:.42)
    .on('mousemove',(e,d0)=>showTip(e,
      `<div class="t">${d0.category}</div><div class="r">
       clock ${fmt(d0.clock_pct,1)}% vs array ${fmt(d0.bg_pct,1)}%<br>
       OR ${fmt(d0.odds_ratio,2)} &middot; q ${d3.format('.1e')(d0.q)}</div>`))
    .on('mouseleave',hideTip);
  F2.g.append('line').attr('x1',x2(0)).attr('x2',x2(0)).attr('y1',0)
    .attr('y2',F2.h).attr('stroke',C.muted);
  F2.g.append('text').attr('class','glab').attr('x',F2.w/2).attr('y',F2.h+36)
    .attr('text-anchor','middle').text('log₂ odds ratio vs array');

  const gs=DATA.genes.slice(0,14);
  const F3=frame('#c-genes',330,{l:100,r:22,t:10,b:44});
  const x3=d3.scaleLinear().domain([0,d3.max(gs,g=>g.count||g.n)*1.1]).range([0,F3.w]);
  const y3=d3.scaleBand().domain(gs.map(g=>g.gene)).range([0,F3.h]).padding(.24);
  F3.g.append('g').attr('class','axis').attr('transform',`translate(0,${F3.h})`)
    .call(d3.axisBottom(x3).ticks(5)).call(s=>s.select('.domain').remove());
  F3.g.append('g').attr('class','axis').call(d3.axisLeft(y3).tickSize(0))
    .call(s=>s.select('.domain').remove())
    .call(s=>s.selectAll('text').attr('fill',C.fg).style('font-style','italic')
      .style('font-size','11.5px'));
  F3.g.selectAll('rect').data(gs).join('rect')
    .attr('x',0).attr('y',g=>y3(g.gene)).attr('height',y3.bandwidth())
    .attr('width',g=>x3(g.count||g.n)).attr('rx',3).attr('fill',C.violet);
  F3.g.append('text').attr('class','glab').attr('x',F3.w/2).attr('y',F3.h+36)
    .attr('text-anchor','middle').text('clock CpGs in gene');
};

activate('overview');
window.addEventListener('resize',()=>{
  const cur=d3.select('.tab.active').datum()[0];
  Object.keys(drawn).forEach(k=>delete drawn[k]);
  RENDER[cur](); drawn[cur]=true;});
</script>
</body>
</html>
"""


def main():
    payload = build_payload()
    blob = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    html = HTML.replace("__DATA__", blob)
    (DOCS / "index.html").write_text(html, encoding="utf-8")
    kb = len(html.encode()) / 1024
    print(f"    wrote docs/index.html  ({kb:.0f} KB, fully self-contained)")
    print(f"    {len(payload['species'])} species, {len(payload['clocks'])} samples, "
          f"{len(payload['states'])} states inlined")


if __name__ == "__main__":
    main()
