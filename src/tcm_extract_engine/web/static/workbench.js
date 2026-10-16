/* Workbench presentation; existing schema and QA editors remain available. */
let jobFilter = 'all', refreshing = false, recordData = null, recordPage = 0;
const originalWorkflow = renderWorkflow;
function toast(message) {
  const el = document.createElement('div'); el.className = 'toast'; el.textContent = message;
  $('#toastStack').append(el); setTimeout(() => el.remove(), 6000);
}
window.alert = toast;
function openUpload() { $('#uploadDialog').showModal(); }
function showPage(page) {
  const optimization = page === 'optimization';
  $('#documentsPage').hidden = optimization; $('#optimizationPage').hidden = !optimization;
  $('#navDocuments').classList.toggle('active', !optimization);
  $('#navOptimization').classList.toggle('active', optimization);
  $('#pageBreadcrumb').textContent = optimization ? '提示词优化' : '文档工作空间';
  if (optimization) loadGepa(true);
}
function confirmAction(message) {
  return new Promise(resolve => {
    const dialog = $('#confirmDialog'); $('#confirmMessage').textContent = message;
    let accepted = false;
    $('#confirmAccept').onclick = () => { accepted = true; dialog.close(); };
    $('#confirmCancel').onclick = () => dialog.close();
    dialog.onclose = () => resolve(accepted); dialog.showModal();
  });
}
drop.onkeydown = e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); } };
uploadFiles = async function(files) {
  const selected = Array.from(files); if (!selected.length || drop.dataset.busy) return;
  drop.dataset.busy = '1'; $('#closeUpload').disabled = true;
  try {
    for (const [i, file] of selected.entries()) {
      $('#uploadStatus').textContent = `正在导入 ${i + 1}/${selected.length}：${file.name}`;
      const form = new FormData(); form.append('file', file); form.append('domain', $('#domain').value);
      form.append('auto_run', String($('#autoRun').checked));
      const response = await fetch('/api/upload?interactive=true', {method:'POST', body:form});
      const data = await response.json(); if (!response.ok) throw new Error(data.detail || '导入失败');
      await refresh(); selectJob(data.job_id);
    }
    $('#uploadStatus').textContent = '导入完成'; $('#uploadDialog').close(); toast('文档已导入');
  } catch (error) { $('#uploadStatus').textContent = error.message; }
  finally { delete drop.dataset.busy; $('#closeUpload').disabled = false; fileInput.value = ''; }
};
function needsAttention(j) { return j.status === 'FAILED' || j.workflow_stage === 'prompt_ready'; }
function setJobFilter(filter) {
  jobFilter = filter;
  document.querySelectorAll('[data-filter]').forEach(el => el.classList.toggle('active', el.dataset.filter === filter));
  renderJobs();
}
refresh = async function() {
  if (refreshing) return; refreshing = true;
  try {
    const response = await fetch('/api/interactive/jobs'); if (!response.ok) throw new Error('服务暂不可用');
    jobs = await response.json();
    $('#connectionStatus').classList.remove('offline'); $('#connectionStatus').innerHTML = '<span class="status-dot"></span>已连接工作空间';
    $('#metricDocuments').textContent = jobs.length;
    $('#metricRunning').textContent = jobs.filter(j => j.status !== 'FAILED' && RUNNING.has(j.workflow_stage)).length;
    $('#metricRecords').textContent = jobs.reduce((n,j) => n + (j.records_count || 0), 0).toLocaleString();
    $('#metricAttention').textContent = jobs.filter(needsAttention).length;
    $('#docCount').textContent = jobs.length; $('#navDocCount').textContent = jobs.length;
    current = jobs.find(j => j.job_id === current?.job_id) || jobs[0] || null;
    renderJobs(); if (current) renderWorkflow();
  } catch (error) {
    $('#connectionStatus').classList.add('offline'); $('#connectionStatus').textContent = '连接中断 · 自动重连';
  } finally { refreshing = false; }
};
selectJob = function(id) {
  current = jobs.find(j => j.job_id === id) || null;
  renderJobs(); if (current) renderWorkflow();
};
renderJobs = function() {
  const query = $('#jobSearch').value.trim().toLowerCase();
  const visible = jobs.filter(j => `${j.filename} ${j.job_id}`.toLowerCase().includes(query)).filter(j =>
    jobFilter === 'all' || (jobFilter === 'running' && j.status !== 'FAILED' && RUNNING.has(j.workflow_stage)) ||
    (jobFilter === 'attention' && needsAttention(j)) || (jobFilter === 'verified' && ['verified','done'].includes(j.workflow_stage)));
  $('#jobs').innerHTML = '';
  if (!visible.length) $('#jobs').innerHTML = '<div class="empty-state">没有匹配的文档</div>';
  visible.forEach(j => {
    const el = document.createElement('button'); el.type = 'button'; el.className = 'job' + (current?.job_id === j.job_id ? ' sel' : '');
    el.setAttribute('aria-pressed', String(current?.job_id === j.job_id));
    el.innerHTML = `<div class="job-top"><span class="file-type">${/\.pdf$/i.test(j.filename) ? 'PDF' : 'MD'}</span><div class="job-copy"><strong>${esc(j.filename)}</strong><small>${esc(j.domain_label || '知识抽取')}</small></div></div><div class="job-bottom"><span class="badge ${badgeClass(j)}">${j.status === 'FAILED' ? '需要重试' : esc(STAGE_LABEL[j.workflow_stage] || j.workflow_stage)}</span><span class="job-date">${j.records_count || 0} 条记录</span></div>`;
    el.onclick = () => selectJob(j.job_id); $('#jobs').append(el);
  });
};
renderStepper = function(stage) {
  const cur = current?.status === 'FAILED' ? ({parse:0,prompt:1,extract:2,verify:4,export:5}[current.failed_stage] ?? STEP_IDX[stage] ?? 0) : (STEP_IDX[stage] ?? 0);
  $('#stepper').innerHTML = STEPS.map((step,i) => `${i ? `<div class="step-line ${i <= cur ? 'done' : ''}"></div>` : ''}<button type="button" class="step" onclick="inspectStep('${step.key}')" ${i > cur ? 'disabled' : ''} title="查看${step.label}"><span class="step-dot ${i < cur ? 'done' : i === cur ? (current.status === 'FAILED' ? 'failed' : 'current') : ''}">${i < cur ? '✓' : i+1}</span><span class="step-label">${step.label}</span></button>`).join('');
};
function inspectStep(key) {
  if (key === 'parse') showParsed();
  else if (key === 'extract' || key === 'compare') showCompare();
  else if (key === 'verify') showVerify();
  else if (key === 'export') showArtifacts();
  else {
    if ($('#planBox')) loadPlan();
    else toast('如需修改抽取方案，请展开高级操作；修改后需重新处理下游步骤。');
  }
}
renderWorkflow = function() {
  const j = current; if (!j) return;
  originalWorkflow();
  $('#wfTitle').textContent = j.filename || j.job_id;
  $('#wfSubtitle').textContent = `${j.domain_label || '知识抽取'} · ${j.chunks_count || 0} 个内容块 · ${j.records_count || 0} 条记录`;
  $('#wfHeaderActions').innerHTML = `${j.records_count ? '<button class="btn" onclick="showArtifacts()">下载数据</button>' : ''}<a class="btn" href="/api/download/${encodeURIComponent(j.job_id)}/pdf" target="_blank" rel="noopener">打开原书 ↗</a>`;
  const panel = $('#panel');
  if (j.status === 'FAILED') {
    const failed = ({profile:'prompt',plan:'prompt',validate:'verify',publish:'export'}[j.failed_stage] || j.failed_stage || 'extract');
    if (panel.dataset.failureKey !== panel.dataset.renderKey || !panel.querySelector('.error-details')) {
      panel.dataset.failureKey = panel.dataset.renderKey;
      const message = String(j.error || j.message || '处理未完成').replace(/^(步骤失败[：:]\s*)+/, '');
      panel.innerHTML = `<div class="stage-head"><div><h3>处理暂停，可以继续</h3><p>重试当前步骤会复用兼容的成功缓存。</p></div></div><div class="workflow-msg error">${esc(STEP_LABEL[failed] || '处理')}未完成</div><details class="error-details"><summary>查看错误详情</summary><pre>${esc(message)}</pre></details><div class="workflow-actions"><button class="btn btn-primary" onclick="post('/api/jobs/${j.job_id}/workflow/retry-${failed}')">重试当前步骤</button>${j.records_count ? '<button class="btn" onclick="showCompare()">查看已有记录</button>' : ''}</div><div id="viewBox"></div>`;
    }
  }
  const progress = j.batch_progress;
  if (j.status !== 'FAILED' && RUNNING.has(j.workflow_stage)) {
    let node = $('#liveProgress');
    if (!node) { node = document.createElement('div'); node.id = 'liveProgress'; panel.prepend(node); }
    const percent = progress?.total ? Math.round(100 * progress.completed / progress.total) : null;
    node.innerHTML = `<div class="progress-track"><div class="progress-fill ${percent === null ? 'indeterminate' : ''}" ${percent !== null ? `style="width:${percent}%"` : ''}></div></div><p class="progress-caption">${percent !== null ? `批次 ${progress.completed}/${progress.total} · ${percent}%` : progress?.cached ? `已保存 ${progress.cached} 个批次缓存，正在继续处理` : '正在处理，请稍候；可切换文档查看其他任务。'}</p>`;
  }
};
controls = function(stage) {
  const key = {parsed:'parse',prompt_ready:'prompt',extracted:'extract',verified:'verify',done:'export'}[stage];
  if (!key) return '';
  return `<details class="advanced-actions"><summary>高级操作</summary><p class="hint">重新处理会更新该步骤及下游状态，请先保存需要的结果。</p><div class="workflow-actions"><button class="btn" onclick="stepClick('${key}')">${RETRY_LABEL[key]}</button><button class="btn" onclick="goBack()">返回上一步</button><button class="btn" onclick="showArtifacts()">下载数据文件</button></div></details>`;
};
showCompare = async function() {
  const id = current.job_id; openView('<div class="workflow-msg info">正在加载记录与证据…</div>');
  try {
    const response = await fetch(`/api/jobs/${id}/compare`); const data = await response.json();
    if (current?.job_id !== id) return; if (!response.ok) throw new Error(data.detail || '加载失败');
    recordData = {...data, jobId:id}; recordPage = 0;
    openView(`<div class="box"><h4>记录与来源证据</h4><div class="record-toolbar"><input id="recordSearch" class="inp" placeholder="搜索主体、内容或记录编号" aria-label="搜索记录" oninput="recordPage=0;renderRecords()"><select id="recordReview" aria-label="复核筛选" onchange="recordPage=0;renderRecords()"><option value="all">全部记录</option><option value="review">待人工复核</option></select></div><div id="recordResults"></div></div>`);
    renderRecords();
  } catch (error) { if (current?.job_id === id) openView(`<div class="workflow-msg error">${esc(error.message)}</div>`); }
};
function recordTitle(r) { return r.subject_text || r.payload?.herb_name || r.payload?.name || r.record_type || r.record_id; }
function renderRecords() {
  if (!recordData || recordData.jobId !== current?.job_id || !$('#recordResults')) return;
  const query = $('#recordSearch').value.toLowerCase();
  const records = (recordData.records || []).filter(r => JSON.stringify(r).toLowerCase().includes(query) && ($('#recordReview').value !== 'review' || r.needs_review));
  const pages = Math.max(1,Math.ceil(records.length/12)); recordPage = Math.min(recordPage,pages-1);
  const visible = records.slice(recordPage*12,recordPage*12+12);
  $('#recordResults').innerHTML = `<p class="result-count">${records.length} 条记录 · 待复核标记不代表已完成准确性确认</p><div class="record-browser"><div class="record-list" id="recordList"></div><div class="record-detail" id="recordDetail">选择一条记录，查看内容与出处。</div></div><div class="record-pagination"><button class="btn" onclick="recordPage--;renderRecords()" ${!recordPage ? 'disabled' : ''}>上一页</button><span>${recordPage+1} / ${pages}</span><button class="btn" onclick="recordPage++;renderRecords()" ${recordPage+1 >= pages ? 'disabled' : ''}>下一页</button></div>`;
  visible.forEach(r => {
    const button = document.createElement('button'); button.className = 'record-row';
    button.innerHTML = `<span class="record-row-title">${esc(recordTitle(r))}</span><span class="record-row-preview">${esc((r.raw_text || JSON.stringify(r.payload || {})).slice(0,85))}</span><span class="record-row-meta">${esc(r.record_type)} ${r.needs_review ? '· 待复核' : ''}</span>`;
    button.onclick = () => { document.querySelectorAll('.record-row').forEach(el => el.classList.remove('active')); button.classList.add('active'); showRecord(r); };
    $('#recordList').append(button);
  });
  $('#recordList button')?.click();
}
function showRecord(record) {
  const ids = record.source?.block_ids || [];
  const chunks = (recordData.chunks || []).filter(c => ids.includes(c.chunk_id)).map(c => ({...c,block_id:c.chunk_id,text:c.content}));
  const labels = {herb_name:'主体名称',image_ref:'关联图片',observations:'图中观察',relations:'关系',steps:'步骤',caption:'图注',description:'描述',statement:'知识陈述',name:'名称',dose:'剂量',interaction:'配伍',pregnancy:'妊娠用药',population:'适用人群',herb_facts:'药物知识',toxicity_level:'毒性'};
  const content = Object.entries(record.payload || {}).filter(([,v]) => v != null && v !== '').map(([key,value]) => `<tr><th>${esc(labels[key] || key)}</th><td>${typeof value === 'object' ? `<pre>${esc(JSON.stringify(value,(k,v) => v === null || ['asset_path','asset_id'].includes(k) ? undefined : v,2))}</pre>` : esc(value)}</td></tr>`).join('');
  const evidence = chunks.map(c => `<div class="source-location">${esc((c.section_path || []).filter(Boolean).join(' / '))} ${c.page != null ? `<a target="_blank" rel="noopener" href="/api/download/${recordData.jobId}/pdf#page=${encodeURIComponent(c.page)}">第 ${esc(c.page)} 页 ↗</a>` : ''} · ${esc(c.block_id)}</div>${c.asset_url ? `<a href="${esc(c.asset_url)}" target="_blank" rel="noopener"><img class="source-image" loading="lazy" src="${esc(c.asset_url)}" alt="该记录关联的原书图片"></a>` : `<blockquote class="source-quote">${esc(c.text || '')}</blockquote>`}`).join('');
  $('#recordDetail').innerHTML = `<h3>${esc(recordTitle(record))}</h3><p class="mono">${esc(record.record_id)}</p><h4>结构化内容</h4><table>${content || '<tr><td>暂无结构化字段</td></tr>'}</table><h4>来源证据</h4>${evidence || '<p class="hint">暂无可展示的来源块，请检查完整记录中的来源信息。</p>'}${record.raw_text ? `<details open><summary>抽取片段 · 请与原书对照</summary><blockquote class="source-quote">${esc(record.raw_text)}</blockquote></details>` : ''}<details><summary>完整记录（JSON）</summary><pre>${esc(JSON.stringify(record,null,2))}</pre></details>`;
}
async function showArtifacts() {
  const id = current.job_id; openView('<p>正在读取可下载文件…</p>');
  try {
    const response = await fetch(`/api/jobs/${id}/artifacts`); const data = await response.json();
    if (id !== current?.job_id) return; if (!response.ok) throw new Error(data.detail || '加载失败');
    openView(`<div class="box"><h4>下载数据文件</h4><p class="hint">仅展示当前有效产物；待复核记录仍需审核。</p><div class="artifact-grid">${data.files.map(f => `<a class="artifact-card" href="${esc(f.url)}"><strong>${esc(f.label)}</strong><small>${esc(f.description)}</small><span>${(f.bytes/1024).toFixed(1)} KB ↓</span></a>`).join('') || '<p>当前步骤尚未生成可下载文件。</p>'}</div></div>`);
  } catch (error) { if (id === current?.job_id) openView(`<p>${esc(error.message)}</p>`); }
}
document.querySelectorAll('[data-icon]').forEach(el => {
  const paths = {plus:'M12 5v14M5 12h14',search:'M21 21l-5-5M18 10a8 8 0 1 1-16 0 8 8 0 0 1 16 0',refresh:'M20 7v5h-5M4 17v-5h5M6 7a7 7 0 0 1 12-2l2 3M4 16l2 3a7 7 0 0 0 12-2',activity:'M2 12h5l3-8 4 16 3-8h5',upload:'M12 16V3m-5 5 5-5 5 5M4 16v5h16v-5',sparkles:'m12 3 3 6 6 3-6 3-3 6-3-6-6-3 6-3Z',flag:'M5 22V3m0 0h14l-3 5 3 5H5',help:'M9 8a3 3 0 1 1 4 3v3m-1 3h.01M22 12a10 10 0 1 1-20 0 10 10 0 0 1 20 0',layers:'m12 3 10 5-10 5L2 8Zm-10 9 10 5 10-5M2 17l10 5 10-5'};
  el.innerHTML = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="${paths[el.dataset.icon] || 'M5 3h14v18H5ZM9 8h6M9 12h6M9 16h4'}"/></svg>`;
});
document.addEventListener('keydown', e => { if ((e.ctrlKey || e.metaKey) && e.key === 'k') { e.preventDefault(); showPage('documents'); $('#jobSearch').focus(); } });
showPage(location.hash === '#optimization' ? 'optimization' : 'documents');
refresh(); setInterval(() => { if (!_fast) refresh(); }, 4000);
