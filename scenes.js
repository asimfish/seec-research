"use strict";
const titles=["棕色小瓶 · 100 ml","透明小瓶 · 100 ml","透明大瓶 · 250 ml","棕色大瓶 · 250 ml"];
async function start(){
 const response=await fetch("data/scene-recovery-audit.json",{cache:"no-cache"});if(!response.ok)throw new Error(`HTTP ${response.status}`);const data=await response.json();
 document.querySelector("#video-list").innerHTML=titles.map((title,i)=>{const n=i+1,a=data.episodes[`precursor_${n}`],phases=i<2?[[0,"初始"],[19,"取出"],[31,"放秤释放"],[35,"再抓"],[48,"归位"]]:[[0,"初始"],[30,"取出"],[55,"放秤释放"],[65,"再抓"],[88,"归位"]];
 return `<article class="video-card"><header>${n}　${title}</header><video controls playsinline preload="metadata" poster="assets/scenes/precursor_${n}.jpg" src="assets/scenes/precursor_${n}.mp4" aria-label="${title}完整物理演示"></video><div class="video-body"><div class="timeline">${phases.map(([time,name])=>`<button type="button" data-time="${time}">${time}s　${name}</button>`).join("")}</div><p>放秤 ${a.placements_mm[0].toFixed(3)} mm · 归位 ${a.placements_mm[1].toFixed(3)} mm · 臂或瓶负分离 ${a.robot_or_bottle_negative_separation_tick_rows} tick</p><p>${a.source_ticks.toLocaleString()} 个物理 tick · 五路各 ${a.full_views.station_a.nb_frames} 帧 / 12 fps</p><a class="small" href="assets/scenes/precursor_${n}.mp4" download>下载保存的站位原片</a></div></article>`;
 }).join("");
 document.querySelectorAll(".video-card").forEach(card=>{const video=card.querySelector("video");card.querySelectorAll("button").forEach(button=>button.addEventListener("click",()=>{const seek=()=>{video.currentTime=Math.min(Number(button.dataset.time),video.duration);};if(video.readyState)seek();else{video.addEventListener("loadedmetadata",seek,{once:true});video.load();}}));});
 const rows=Object.values(data.episodes).reduce((n,e)=>n+e.contact_tick_rows,0);
 document.querySelector("#contact-note").textContent=`本次重新扫描全部 ${rows.toLocaleString()} 行接触记录，臂或瓶负分离均为 0。整个场景仍有静态摆放负分离：每条开头 95 ticks，最深约 −1.751 mm，最后出现约 0.392 s；不能写成“全场景零穿透”。`;
 document.querySelector("#recovery-note").textContent=`本次往返的规范化 JSON 载荷：${data.fresh_serialization_roundtrip_exact?"逐字节相同":"未通过"}。真实旧 adapter 的 CPU setter 探针：保存关节速度 ${data.fresh_adapter_probe.saved_joint_velocity_rad_s} rad/s，恢复写入 ${data.fresh_adapter_probe.restored_joint_velocity_max_rad_s} rad/s。它可用于登记的静止边界协议，不能当作运动中状态精确恢复。局部携行 PASS 不覆盖全场景 FAIL。`;
}
start().catch(error=>{document.querySelector("#video-list").innerHTML='<p role="alert">审计快照读取失败，请打开下方数据与原场景档案。</p>';console.error(error);});
