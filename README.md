# 🏡 🏯 UnderOneFacade 🏢 🏡

**Repo of the ECCV'26 paper "UnderOneFacade: Worldwide Facade Semantic Segmentation Benchmark Dataset"**

[[Dataset]](#) [[Paper]](#) [[Benchmark]](#) [[More]](#)

<!-- 把你的方法图/teaser图放进 img/ 文件夹，文件名换成你自己的 -->
![teaser](img/teaser.png)

## 🌟 Highlights

- **2.7 billion** annotated points — the largest cross-country 3D facade benchmark to date
- Spans **three continents** (UK, Germany, Singapore), combining Victorian, Haussmann, and Southeast Asian colonial/modern architectural styles
- Centimeter-accurate geometry from **multi-sensor acquisition** (TLS + MLS: Leica RTC360, Leica BLK360, MODISSA platform)
- Adopts and extends the **LoFG (Level of Facade Generalization)** taxonomy across countries — 15 classes at LoFG3, 5 at LoFG2
- Benchmarks **6 representative architectures** (PointNet++, KPConv, DGCNN, PointTransformer v1, PointTransformer v3, OctFormer)
- Reveals strong **cross-continental domain shift**: several models degrade by more than 30 F1 points between European and Asian facades

## 📝 Abstract

Globally consistent semantic digital twins require centimeter-accurate and geographically transferable 3D facade segmentation. However, progress in facade parsing is limited by the lack of large-scale, standardized benchmarks for evaluating cross-domain generalization. Existing datasets are geographically narrow, semantically inconsistent, or insufficiently precise. We introduce UnderOneFacade, the largest cross-country and cross-continent 3D facade benchmark to date, comprising centimeter-accurate point clouds with hierarchical, harmonized, and architecturally grounded semantic labels totaling 2.7 billion annotated points. Through a systematic evaluation of representative point-, graph- and transformer-based architectures, we show that current methods struggle to recognize fine-grained architectural elements and degrade significantly across geographic domains, with the best models achieving only up to 33 IoU on the fine-grained LoFG3 benchmark. By combining geometric precision with standardized semantics at unprecedented scale, UnderOneFacade establishes a rigorous benchmark for developing robust and transferable 3D segmentation models. The dataset, evaluation scripts, and pretrained models will be released upon publication.

## 🎓 Publication

Please find our paper accepted at ECCV 2026:


## 🤝 Acknowledgments
