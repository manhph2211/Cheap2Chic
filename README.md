
<div align="center" style="font-size: 5em;">
  <strong>From Cheap to Chic: Enhancing Music Playback Quality of Budget Earphones via Hardware-Aware Learning (Sensys 2025)</strong>
  <br> </br> 
</div>

<div align="center"> 
<a href="https://manhph2211.github.io/Cheap2Chic/"><img src="https://img.shields.io/badge/Website-Cheap2Chic WebPage-blue?style=for-the-badge"></a>
<a href=""><img src="https://img.shields.io/badge/arxiv-Paper-red?style=for-the-badge"></a>
<a href=""><img src="https://img.shields.io/badge/Checkpoint-%F0%9F%A4%97%20Hugging%20Face-White?style=for-the-badge"></a>
</div>

<div align="center">
  <a href="https://laserhu.github.io/" target="_blank">Hu&nbsp;Changshuo*</a> &emsp;
  <a href="https://maxph2211.dev/" target="_blank">Hung&nbsp;Manh&nbsp;Pham*</a> &emsp;
  <a href="" target="_blank">Ting&nbsp;Dang</a> &emsp;
  <a href="" target="_blank">Jiannan&nbsp;Li</a> &emsp;
  <a href="" target="_blank">Rajesh&nbsp;Balan</a> &emsp;
  <a href="https://www.dongma.info/" target="_blank">Dong&nbsp;Ma</a> &emsp;
</div>
<br>

<!-- <div align="center">
    <img src="" alt="Illustration of our technique"/>
</div> -->

## :rocket: Introduction

Low-end earphones are widely spread due to their affordability, but their limited speaker hardware often leads to poor music playback quality. This raises a key question: **Can we compensate for hardware limitations to enhance the listening experience without modifying the device?**

In this work:
- We proposed Cheap2Chic, a novel framework that enhances music delivery quality on low-end earphones. Cheap2Chic innovates with (1) a deep learning-based approach for speaker characterization and (2) a cascaded architecture that learns effective compensation strategies to address speaker deficiencies.

- We designed a data synthesis module that accurately simulates earphone recordings by leveraging coarse-grained FRC data. This approach innovatively treats the FRC as an image and extracts features using a vision-language model (VLM).

- We built a prototype and collected a dataset of playback recordings from eight commercial earphones across ten music genres (totaling 40 hours). We plan to release this dataset publicly to facilitate and encourage broader research on earphone audio quality.

- We conducted extensive experiments to evaluate the effectiveness and system overhead of Cheap2Chic under various real-world factors.