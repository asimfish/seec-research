"use strict";

const UI = {
  en: {
    nav: ["Overview", "Experiments", "Roadmap", "Resources"], brand: "Research journal", progress: "In progress",
    eyebrow: "Student-executable experience consolidation", hero: "Experience should be<br><em>executable by the student.</em>",
    lead: "A successful teacher can use information the student never sees. SEEC studies how to audit and route those demonstrations before learning, and what evidence is needed to trust the resulting policy.",
    paper: "Read the paper", explore: "Explore experiments", dated: "Updated", draft: "ICML-style draft · research in progress",
    snapshot: "Research snapshot", deferred: "Main study deferred", pilot: "Original neural pilot", pilotNote: "Selected held-out episodes; five paired training seeds.",
    feedback: "Known-model feedback witness", feedbackNote: "Hand-designed control on reused conditions, not learned-policy benefit.", records: "Latest diagnostic records", recordsNote: "Five preregistered controls; every failure retained.",
    method: "The method, at the student's boundary", methodText: "Check what the student can actually observe. Bind evidence to the actions that were executed. Keep routing claims separate from learned outcomes.", zoom: "Click a figure to inspect it", methodCaption: "Figure 1 · Problem, source-bound evidence, routing and the separate efficacy question.",
    steps: [["Canonicalize the input", "Exact ten-frame student windows define identical-input classes."], ["Check source-bound evidence", "Registered candidate paths and probes are rebuilt from source anchors."], ["Route the targets", "Executable, INFO, recovery or escalation. Recovery remains untested here."], ["Evaluate a fixed learner", "Teacher-free terminal outcomes require their own paired evidence."]],
    boundary: "Certificates cover finite registered candidates, not every continuous action or feedback policy. The controlled carrier is engineered to expose the conflict. Technical checks do not establish a learning advantage.",
    latest: "Latest finding", latestTitle: "Fixed-path failure does not rule out visible feedback.", latestText: "With the same source archive, model and tolerance, half-mean excitation followed by observed-gain correction succeeds in 120/120 conditions. Nominal gain succeeds in 40/120. Full scaling retains 18 saturation failures.", latestScope: "A constructive known-model witness, not a new adaptive-control algorithm or a successful neural SEEC policy.", latestLink: "Inspect all five controls →",
    openTitle: "What is still unresolved?", roadmapLink: "View the research roadmap →", allExperiments: "All experiments →",
    experimentsTitle: "Experiments, with their boundaries.", experimentsIntro: "Each study has its own data, unit and stopping rule. Successful controls, negative learner results and technical audits are reported without pooling them into a single score.",
    categories: {all: "All studies", learning: "Learning", diagnostic: "Mechanism", audit: "Audit", reference: "References"}, search: "Search studies…", searchLabel: "Search experiments", empty: "No studies match this filter.", studyLink: "Source / report ↗",
    chartTitle: "Observable-feedback diagnostic", chartIntro: "The five controls were frozen before execution. Only the gain-estimation rule or excitation changes; dynamics and terminal tolerance stay fixed.",
    scopes: {all: "All reused conditions", source: "Source conditions", diagnostic_reused: "Diagnostic conditions"},
    controlLabels: {raw_retrieval: "Raw nearest-class mean", half_mean_observed: "Half mean / observed gain", half_mean_nominal: "Half mean / nominal gain", full_mean_observed: "Full mean / observed gain", zero_observed: "Zero excitation"},
    chartFoot: "Counts are conditions, not independent training seeds. No confidence interval or generalization claim. 600 records / 7800 simulation steps / zero training updates. One verifier/evaluator threshold disagreement is retained.",
    gallery: "The manuscript figure library", galleryIntro: "18 source-linked figures. Open the vector original for details; captions retain the scope of the corresponding experiment.", figure: "Figure", imageOpen: "Open figure", vector: "Vector", download: "Download PDF",
    roadmapTitle: "Progress is an evidence chain.", roadmapIntro: "The next milestone is successful neural closed-loop execution under the declared student boundary, followed by a comparison that identifies the value of evidence against strong certificate-free rivals.", milestones: "Current milestones", history: "Research record", historyText: "Failed versions and post-result method revisions remain part of the record. These entries are milestones, not independent replications.", next: "Next scientific questions", noReadiness: "Main efficacy is deferred. This page does not claim conference readiness, natural-data non-inferiority, real-robot success, VLA performance or a budget advantage.",
    resourcesTitle: "A readable record. A checkable snapshot.", resourcesIntro: "The public package contains the current manuscript, figures and curated result summaries. The underlying research repository and unselected artifacts remain private.",
    resourceItems: [["PAPER", "Current manuscript", "ICML-style draft, including negative pilot results, candidate-scope limitations and Appendix R's feedback witness.", "Open PDF →", "assets/seec-paper.pdf"], ["DATA", "Latest diagnostic summary", "All five outcomes, complete coverage, numerical disagreement, execution identity and record-chain head.", "Read JSON →", "data/feedback-summary.json"], ["RECORD", "Public progress snapshot", "Study scopes, milestones, figures, results and the paper identity behind this page.", "Read JSON →", "data/progress.json"], ["MANIFEST", "Publication manifest", "SHA-256 identities for every exported asset, and explicit public-export boundaries.", "Read manifest →", "publication-manifest.json"], ["CHECKS", "Scoped delivery verification", "18 artifact/protocol checks. They do not certify scientific efficacy or conference readiness.", "Read verification →", "data/delivery-verification.json"], ["GITHUB", "Website source", "A standalone static site with no external runtime dependencies. Separate from the private scientific repository.", "Open repository ↗", "repository"]],
    provenance: "Snapshot provenance", revision: "Research revision", freeze: "Diagnostic freeze", plan: "Diagnostic plan", paperHash: "Paper SHA-256", chain: "Record-chain head", checked: "Artifact checks", paperFormat: "Paper layout", layout: "8-page body + Impact; references begin on page 9; 26 pages including appendices.", designReferences: "Presentation references", credits: "Layout inspiration only. No benchmark result, media asset or code from these references is presented as our research.", footer: "Source-bound research on demonstrations, information and learnability.", footerNote: "Research in progress. Audit success, model-based feasibility and learned-policy efficacy are different claims.", states: {complete:"Recorded",checked:"Checked",stopped:"Stopped",open:"Open",deferred:"Deferred"}, reportError: "The progress snapshot could not load. Reload this page or open the paper PDF directly.", close: "Close figure"
  },
  zh: {
    nav: ["方案概览", "实验记录", "进展路线", "论文与资料"], brand: "科研进展记录", progress: "研究进行中",
    eyebrow: "Student-executable experience consolidation", hero: "让经验固化为<br><em>学生真正能执行的策略。</em>",
    lead: "教师完成任务时，可能用到了学生看不到的信息。SEEC 研究如何在学习前审计并路由这些示范，以及什么样的证据才能支撑最终策略的有效性。",
    paper: "阅读论文", explore: "查看实验", dated: "更新于", draft: "ICML 格式草稿 · 研究进行中",
    snapshot: "当前研究状态", deferred: "主实验暂缓", pilot: "原神经学习器 pilot", pilotNote: "选定检查点的留出 episode；五个配对训练种子。",
    feedback: "已知模型反馈见证", feedbackNote: "复用条件下的手工控制，不是学得的策略收益。", records: "最新诊断记录", recordsNote: "五个事前登记的控制器，全部失败保留。",
    method: "从学生真正可见的边界出发", methodText: "检查学生能观察到什么，将证据绑定到真实执行动作，并把路由结论与学习器效果分开验证。", zoom: "点击图表可放大查看", methodCaption: "图 1 · 问题、源绑定证据、路由，以及单独验证的效果问题。",
    steps: [["规范化学生输入", "精确的十帧观测窗口定义同输入类。"], ["核对源绑定证据", "从源锚点重建登记的候选路径与 probe 证据。"], ["路由学习目标", "可执行、INFO、恢复或升级；本实验尚未验证恢复。"], ["评测固定学习器", "无教师的终局成功需要独立的配对实证。"]],
    boundary: "证书只覆盖登记的有限候选，不能排除所有连续动作或反馈策略。受控载体为展示冲突而构造，技术检查通过并不等于学习收益成立。",
    latest: "最新发现", latestTitle: "固定路径失败，不等于可见反馈不可行。", latestText: "在相同源档案、模型与容差下，半幅均值激励加可见增益修正成功 120/120；名义增益对照成功 40/120。完整均值保留了 18 条限幅失败。", latestScope: "这是已知模型的构造性见证，不是新的自适应控制算法，也不是成功的神经 SEEC 策略。", latestLink: "查看全部五个对照 →",
    openTitle: "还有哪些核心问题？", roadmapLink: "查看研究路线 →", allExperiments: "全部实验 →",
    experimentsTitle: "完整记录结果，也完整说明边界。", experimentsIntro: "每项研究都有自己的数据、统计单位与停止规则。成功控制、阴性学习器结果与技术审计单独报告，不合并成一个总分。",
    categories: {all: "全部研究", learning: "学习器", diagnostic: "机制诊断", audit: "审计", reference: "参考实验"}, search: "搜索实验…", searchLabel: "搜索实验记录", empty: "没有符合筛选条件的研究。", studyLink: "来源 / 报告 ↗",
    chartTitle: "可见反馈诊断", chartIntro: "五个控制器均在执行前冻结，只改变增益估计规则或激励，动力学与终局容差保持不变。",
    scopes: {all: "全部复用条件", source: "源条件", diagnostic_reused: "诊断条件"},
    controlLabels: {raw_retrieval: "原始最近类均值", half_mean_observed: "半幅均值 / 可见增益", half_mean_nominal: "半幅均值 / 名义增益", full_mean_observed: "完整均值 / 可见增益", zero_observed: "零激励"},
    chartFoot: "计数单位是条件，不是独立训练种子。不附置信区间，不作泛化主张。600 条记录 / 7800 仿真步 / 零训练更新，保留 1 条验证器与评测器的阈值分歧。",
    gallery: "论文图表库", galleryIntro: "18 张关联源记录的图表。可打开矢量原图查看细节，图注保留对应实验的适用范围。", figure: "图", imageOpen: "打开图表", vector: "矢量图", download: "下载 PDF",
    roadmapTitle: "进展是一条逐项关闭的证据链。", roadmapIntro: "下一里程碑是在登记的学生边界下完成神经策略闭环成功，再设计能够区分证书与强无证书方法的比较。", milestones: "当前里程碑", history: "研究记录", historyText: "失败的版本与看到结果后的方法修订均保留。这些条目是研究里程碑，不是独立重复实验。", next: "下一步科学问题", noReadiness: "主效果实验暂缓。本页面不主张顶会就绪、自然数据非劣、真机成功、VLA 性能或预算优势。",
    resourcesTitle: "读得清楚，也能核对。", resourcesIntro: "公开包包含当前论文、图表与精选结果摘要。原科研仓库及未选入的产物保持私有。",
    resourceItems: [["PAPER", "当前论文", "ICML 格式草稿，包含阴性 pilot、候选范围限制与附录 R 的反馈见证。", "打开 PDF →", "assets/seec-paper.pdf"], ["DATA", "最新诊断摘要", "五个对照的完整结局、覆盖、数值分歧、执行身份与记录链头。", "阅读 JSON →", "data/feedback-summary.json"], ["RECORD", "公开进展快照", "研究范围、里程碑、图表、结果与本页面对应的论文身份。", "阅读 JSON →", "data/progress.json"], ["MANIFEST", "发布清单", "全部导出文件的 SHA-256 身份与明确的公开导出边界。", "阅读清单 →", "publication-manifest.json"], ["CHECKS", "有范围的交付验证", "18 项产物 / 协议检查，不能认证科学效果或顶会就绪。", "阅读验证 →", "data/delivery-verification.json"], ["GITHUB", "网站源码", "无外部运行依赖的独立静态站点，与私有科研仓库分开。", "打开仓库 ↗", "repository"]],
    provenance: "快照来源", revision: "科研提交", freeze: "诊断冻结提交", plan: "诊断计划", paperHash: "论文 SHA-256", chain: "记录链头", checked: "产物检查", paperFormat: "论文版式", layout: "正文与 Impact 共 8 页；参考文献从第 9 页开始；含附录共 26 页。", designReferences: "展示形式参考", credits: "仅参考信息布局；不把这些参考项目的结果、媒体或代码当作我们的科研产物。", footer: "围绕示范、信息边界与可学习性的源绑定研究。", footerNote: "研究进行中。审计通过、模型控制可行与学得策略有效是不同的主张。", states: {complete:"已记录",checked:"已核对",stopped:"已停止",open:"未关闭",deferred:"暂缓"}, reportError: "进展快照加载失败，请刷新页面，或直接打开论文 PDF。", close: "关闭图表"
  }
};

const VIEWS = ["overview", "experiments", "roadmap", "resources"];
let data;
let lang = "en";
let view = "overview";
let category = "all";
let search = "";
let chartScope = "all";
const esc = value => String(value).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const tr = object => object[lang] || object.en;
const badge = state => `<span class="badge ${esc(state)}">${esc(UI[lang].states[state])}</span>`;
const internal = (target, text, classes="link-arrow") => `<button class="${classes}" type="button" data-view="${target}">${esc(text)}</button>`;
const heading = (id, number, title, description) => `<div class="section-head"><div><div class="section-number">${number}</div><h2 id="${id}">${esc(title)}</h2><p>${esc(description)}</p></div></div>`;
const questions = () => `<ul class="compact-list">${data.questions.map((q,i)=>`<li><span class="index">0${i+1}</span><div><b>${esc(tr(q.title))}</b><p>${esc(tr(q.text))}</p></div></li>`).join("")}</ul>`;
const figureButton = (id, title, thumb=false) => `<button type="button" class="figure-button" data-figure="${esc(id)}" aria-label="${esc(UI[lang].imageOpen+": "+title)}"><img src="assets/figures/${esc(id)}.${thumb?"png":"svg"}" alt="${esc(title)}" ${thumb?'loading="lazy"':'fetchpriority="high"'} width="1100" height="500"></button>`;

function renderOverview() {
  const u=UI[lang];
  document.getElementById("overview").innerHTML=`<div class="container"><div class="hero"><div><div class="eyebrow">${u.eyebrow}</div><h1 id="overview-title">${u.hero}</h1><p class="lead">${u.lead}</p><div class="actions"><a class="button primary" href="assets/seec-paper.pdf">${u.paper} <span aria-hidden="true">↗</span></a>${internal("experiments",u.explore,"button")}</div><div class="hero-meta">${u.dated} ${data.updated} · ${u.draft}</div></div><div class="hero-card"><div class="card-heading"><span>${u.snapshot}</span><span class="badge deferred">${u.deferred}</span></div><div class="metric"><div><div class="metric-label">${u.pilot}</div><div class="metric-note">${u.pilotNote}</div></div><div class="metric-value negative">0/1715</div></div><div class="metric"><div><div class="metric-label">${u.feedback}</div><div class="metric-note">${u.feedbackNote}</div></div><div class="metric-value">120/120</div></div><div class="metric"><div><div class="metric-label">${u.records}</div><div class="metric-note">${u.recordsNote}</div></div><div class="metric-value">600</div></div></div></div><div class="status-strip">${data.milestones.slice(1).map(m=>`<div class="status-item"><b>${esc(tr(m.title))}</b> ${badge(m.state)}<p>${esc(tr(m.text))}</p></div>`).join("")}</div><section class="section">${heading("method-title","01 / METHOD",u.method,u.methodText)}<div class="method-figure">${figureButton("seec_overview",u.method)}<div class="figure-foot"><span>${u.methodCaption}</span><span>${u.zoom} ↗</span></div></div><div class="method-steps">${u.steps.map((s,i)=>`<div class="method-step"><span>0${i+1}</span><h3>${s[0]}</h3><p>${s[1]}</p></div>`).join("")}</div><div class="notice">${u.boundary}</div></section><section class="section split"><article class="finding"><div class="date">2026-10-10 / ${u.latest}</div><h3>${u.latestTitle}</h3><p>${u.latestText}</p><p>${u.latestScope}</p>${internal("experiments",u.latestLink)}</article><div><h2>${u.openTitle}</h2>${questions()}${internal("roadmap",u.roadmapLink)}</div></section></div>`;
}

function studyCards() {
  const u=UI[lang];
  const studies=data.studies.filter(s=>(category==="all"||s.category===category)&&[tr(s.title),tr(s.description),tr(s.outcome),tr(s.scope),s.id].join(" ").toLowerCase().includes(search.toLowerCase()));
  document.getElementById("study-grid").innerHTML=studies.length?studies.map(s=>`<article class="study-card" data-study="${esc(s.id)}"><div class="study-top"><span>${esc(u.categories[s.category])}</span>${badge(s.state)}</div><h3>${esc(tr(s.title))}</h3><p>${esc(tr(s.description))}</p><div class="study-outcome">${esc(tr(s.outcome))}</div><div class="study-scope">${esc(tr(s.scope))}</div><a class="study-link" href="${esc(s.report)}">${u.studyLink}</a></article>`).join(""):`<p class="empty-state" role="status">${u.empty}</p>`;
  document.querySelectorAll(".filter-tab").forEach(b=>b.setAttribute("aria-pressed",String(b.dataset.filter===category)));
}

function renderChart() {
  const u=UI[lang];
  const rows=Object.entries(data.feedback.outcomes).map(([policy,parts])=>{const selected=chartScope==="all"?Object.values(parts):[parts[chartScope]]; return {policy,success:selected.reduce((a,p)=>a+p.success,0),total:selected.reduce((a,p)=>a+p.total,0)};});
  const order=["raw_retrieval","half_mean_observed","half_mean_nominal","full_mean_observed","zero_observed"];
  rows.sort((a,b)=>order.indexOf(a.policy)-order.indexOf(b.policy));
  document.getElementById("bar-chart").innerHTML=rows.map(r=>`<div class="bar-row"><span class="bar-label">${esc(u.controlLabels[r.policy])}</span><div class="bar-track" role="img" aria-label="${esc(u.controlLabels[r.policy])}: ${r.success}/${r.total}"><div class="bar-fill ${r.policy==="half_mean_observed"?"main":""}" style="width:${100*r.success/r.total}%"></div></div><strong>${r.success}/${r.total}</strong></div>`).join("");
  document.querySelectorAll(".scope-tab").forEach(b=>b.setAttribute("aria-pressed",String(b.dataset.scope===chartScope)));
}

function renderExperiments() {
  const u=UI[lang];
  document.getElementById("experiments").innerHTML=`<div class="container"><div class="page-heading"><div class="eyebrow">Research / experiments</div><h1 id="experiments-title">${u.experimentsTitle}</h1><p>${u.experimentsIntro}</p></div><div class="toolbar"><div class="filter-tabs" role="group" aria-label="${lang==="en"?"Study category":"研究类别"}">${Object.entries(u.categories).map(([key,label])=>`<button type="button" class="filter-tab" data-filter="${key}" aria-pressed="${key===category}">${label}</button>`).join("")}</div><input id="study-search" class="search" type="search" placeholder="${u.search}" aria-label="${u.searchLabel}" value="${esc(search)}"></div><div id="study-grid" class="study-grid"></div><section class="chart-panel" aria-labelledby="chart-title"><div class="chart-title"><div><h2 id="chart-title">${u.chartTitle}</h2><p>${u.chartIntro}</p></div><div class="scope-tabs" role="group" aria-label="${lang==="en"?"Condition scope":"条件范围"}">${Object.entries(u.scopes).map(([key,label])=>`<button type="button" class="scope-tab" data-scope="${key}" aria-pressed="${key===chartScope}">${label}</button>`).join("")}</div></div><div class="bar-chart" id="bar-chart"></div><div class="chart-axis"><span>0%</span><span>50%</span><span>100%</span></div><p class="chart-foot">${u.chartFoot}</p></section><section class="section">${heading("gallery-title","18 / FIGURES",u.gallery,u.galleryIntro)}<div class="gallery-grid">${data.figures.map(f=>`<article class="gallery-card">${figureButton(f.id,tr(f.title),true)}<div class="gallery-caption"><span class="badge">${u.figure} ${f.number}</span><h3>${esc(tr(f.title))}</h3><p>${esc(tr(f.caption))}</p><div class="gallery-links"><a href="assets/figures/${f.id}.svg">${u.vector} ↗</a><a href="assets/figures/${f.id}.pdf">PDF ↗</a></div></div></article>`).join("")}</div></section></div>`;
  studyCards();renderChart();
}

function renderRoadmap() {
  const u=UI[lang];
  document.getElementById("roadmap").innerHTML=`<div class="container"><div class="page-heading"><div class="eyebrow">Research / roadmap</div><h1 id="roadmap-title">${u.roadmapTitle}</h1><p>${u.roadmapIntro}</p></div><section class="section"><h2>${u.milestones}</h2><div class="milestones">${data.milestones.map((m,i)=>`<article class="milestone"><div class="step-number">0${i+1} / EVIDENCE</div>${badge(m.state)}<h3>${esc(tr(m.title))}</h3><p>${esc(tr(m.text))}</p></article>`).join("")}</div><div class="notice amber">${u.noReadiness}</div></section><section class="section">${heading("history-title","RECORD / HISTORY",u.history,u.historyText)}<div class="timeline">${data.timeline.map(t=>`<article class="timeline-entry"><div class="timeline-date">${esc(t.date)}</div><h3>${esc(tr(t.title))}</h3><p>${esc(tr(t.text))}</p></article>`).join("")}</div></section><section class="section"><h2>${u.next}</h2>${questions()}</section></div>`;
}

function renderResources() {
  const u=UI[lang]; const p=data.provenance;
  const entries=[[u.revision,p.research_revision],[u.freeze,p.freeze_revision],[u.plan,p.plan_id],[u.paperHash,p.paper_sha256],[u.chain,p.chain_head],[u.checked,`${p.delivery_passed}/${p.delivery_total} · artifact/protocol only`],[u.paperFormat,u.layout]];
  document.getElementById("resources").innerHTML=`<div class="container"><div class="page-heading"><div class="eyebrow">Research / resources</div><h1 id="resources-title">${u.resourcesTitle}</h1><p>${u.resourcesIntro}</p></div><div class="resources-grid">${u.resourceItems.map(r=>`<a class="resource" href="${esc(r[4]==="repository"?data.repository_url:r[4])}"><div class="resource-icon">${r[0]}</div><h3>${r[1]}</h3><p>${r[2]}</p><span>${r[3]}</span></a>`).join("")}</div><section class="section provenance"><h2>${u.provenance}</h2><dl>${entries.map(([name,value])=>`<dt>${esc(name)}</dt><dd><code>${esc(value)}</code></dd>`).join("")}</dl></section><section class="section"><h3>${u.designReferences}</h3><p>${u.credits}</p><ul>${data.references.map(r=>`<li><a href="${esc(r.url)}" target="_blank" rel="noopener noreferrer">${esc(r.name)} ↗</a></li>`).join("")}</ul></section></div>`;
}

function readLocation() {
  const p=new URLSearchParams(location.search);
  lang=["en","zh"].includes(p.get("lang"))?p.get("lang"):(navigator.language.startsWith("zh")?"zh":"en");
  view=VIEWS.includes(p.get("view"))?p.get("view"):"overview";
}

function updateLocation(push=true) {
  const url=new URL(location.href);url.searchParams.set("lang",lang);url.searchParams.set("view",view);
  if(push)history.pushState({},"",url);else history.replaceState({},"",url);
}

function render() {
  const u=UI[lang]; document.documentElement.lang=lang==="zh"?"zh-CN":"en";
  document.title=`SEEC · ${u.nav[VIEWS.indexOf(view)]} · ${u.progress}`;
  document.getElementById("navigation").innerHTML=VIEWS.map((v,i)=>`<button type="button" data-view="${v}" class="nav-button" ${v===view?'aria-current="page"':""}>${u.nav[i]}</button>`).join("");
  document.querySelector(".brand").href=`?lang=${lang}&view=overview`;
  document.getElementById("brand-subtitle").textContent=u.brand;
  document.getElementById("header-state").textContent=u.progress;
  const languageButton=document.getElementById("language");languageButton.textContent=lang==="en"?"中文":"EN";languageButton.setAttribute("aria-label",lang==="en"?"切换到中文":"Switch to English");
  renderOverview();renderExperiments();renderRoadmap();renderResources();
  VIEWS.forEach(v=>document.getElementById(v).hidden=v!==view);
  document.getElementById("footer-description").textContent=u.footer;
  document.getElementById("footer-links").innerHTML=`<div class="footer-links"><a href="assets/seec-paper.pdf">${u.paper} ↗</a><a href="${esc(data.repository_url)}">GitHub ↗</a><a href="publication-manifest.json">Manifest ↗</a></div>`;
  document.getElementById("footer-meta").textContent=`${u.dated} ${data.updated} · ${u.footerNote}`;
  document.getElementById("close-dialog").setAttribute("aria-label",u.close);
}

document.addEventListener("click",event=>{
  const target=event.target.closest("button");if(!target||!data)return;
  if(target.dataset.view){view=target.dataset.view;updateLocation();render();window.scrollTo({top:0,behavior:"instant"});}
  if(target.id==="language"){lang=lang==="en"?"zh":"en";updateLocation();render();}
  if(target.dataset.filter){category=target.dataset.filter;studyCards();}
  if(target.dataset.scope){chartScope=target.dataset.scope;renderChart();}
  if(target.dataset.figure){const f=data.figures.find(x=>x.id===target.dataset.figure);if(!f)return;document.getElementById("dialog-title").textContent=`${UI[lang].figure} ${f.number} · ${tr(f.title)}`;document.getElementById("dialog-image").src=`assets/figures/${f.id}.svg`;document.getElementById("dialog-image").alt=tr(f.title);document.getElementById("dialog-caption").textContent=tr(f.caption);document.getElementById("figure-dialog").showModal();}
  if(target.id==="close-dialog")document.getElementById("figure-dialog").close();
});
document.addEventListener("input",event=>{if(event.target.id==="study-search"){search=event.target.value;studyCards();}});
window.addEventListener("popstate",()=>{readLocation();if(data)render();});

async function init() {
  readLocation();
  try {const response=await fetch("data/progress.json");if(!response.ok)throw new Error("snapshot unavailable");data=await response.json();render();updateLocation(false);document.getElementById("load-state").hidden=true;}
  catch {const node=document.getElementById("load-state");node.textContent=UI[lang].reportError;const link=document.createElement("a");link.href="assets/seec-paper.pdf";link.textContent=UI[lang].paper;node.append(document.createElement("br"),link);}
}
init();
