---
layout: default
title: FACTS - Femur Archaeological CT Superresolution
---

<figure>
  <img class="figure" src="https://github.com/phialosophy10/BoneSuperResolution/assets/93533251/945d4ad4-9023-4e59-9b42-a36c5e1b2978" alt="3D rendering of a micro-CT scan of a proximal femur with the femoral neck highlighted">
  <figcaption>3D rendering of one of the micro-CT scans, with the femoral neck highlighted.</figcaption>
</figure>

Super-resolution (SR) methods are usually developed and tested on low-resolution images that were made by downsampling high-resolution images. Real clinical scans are degraded in more complex ways. **FACTS** contains real pairs instead: each bone was scanned with a clinical CT scanner and with a micro-CT scanner, and the two scans were registered so the voxels correspond. The dataset was first published with the paper [*Superresolution of real-world multiscale bone CT verified with clinical bone measures*](https://doi.org/10.1007/978-3-031-66958-3_12) (MIUA 2024).

This page introduces the dataset, describes the preprocessing used for the benchmark, and reports results for SRGAN and ESRGAN trained on real and on synthetic low-resolution data.

## Dataset {#dataset}

FACTS contains 13 archaeological proximal femurs from humans dated around the Middle Ages in Denmark. It includes left and right femurs from both sexes (2 M / 11 F).

| Name  | Sex | Age             | Shape                |
|-------|-----|-----------------|----------------------|
| f_001 | F   | Middle-aged     | [904, 1348, 1561]    |
| f_002 | F   | unknown         | [824, 1492, 1554]    |
| f_013 | F   | unknown         | [976, 1202, 1501]    |
| f_015 | M   | Middle-aged/old | [1474, 1198, 1798]   |
| f_021 | M   | Young           | [1198, 1498, 1697]   |
| f_026 | F   | unknown         | [810, 1478, 1544]    |
| f_074 | F   | Young           | [982, 1618, 1595]    |
| f_075 | F   | unknown         | [846, 1424, 1600]    |
| f_083 | F   | unknown         | [1048, 1344, 1430]   |
| f_086 | F   | unknown         | [716, 1434, 1431]    |
| f_138 | F   | unknown         | [1228, 1226, 1412]   |
| f_164 | F   | unknown         | [914, 1266, 1367]    |
| f_172 | F   | unknown         | [1056, 1410, 1688]   |

### Scanning modalities

* **Clinical CT (LR):** Siemens clinical CT scanner, 0.21 &times; 0.21 &times; 0.4 mm³ voxels.
* **Micro-CT (HR):** Nikon micro-CT scanner, 58 &times; 58 &times; 58 µm³ voxels.

The HR and LR volumes of each bone were registered in ITK-SNAP. The LR volume was then resliced onto the HR grid with linear interpolation, so the two volumes correspond voxel to voxel.

The dataset also contains synthetic low-resolution volumes (SY) and binary bone masks (MS). The masks mark whether a voxel is inside (255) or outside (0) the bone surface. They were made by blurring, dilating and thresholding the HR volumes.

| Modality          | Resolution            | File type | Data type | Voxel intensities (avg.)                       |
|-------------------|-----------------------|-----------|-----------|------------------------------------------------|
| Micro-CT (HR)     | 58 &times; 58 &times; 58 µm³ | TIFF | float32 | min/max: -100 / 1000, mean/median: 15.3 / 1.4 |
| Clinical CT (LR)  | 0.21 &times; 0.21 &times; 0.4 mm³ | TIFF | float32 | min/max: -1000 / 3000, mean/median: -736 / -982 |
| Synthetic (SY)    | -                     | TIFF      | float32   | -                                              |
| Mask (MS)         | -                     | TIFF      | uint8     | -                                              |

### Folder structure

```
FACTS/
├── f_001/
│   ├── HR/f_001.tif        micro-CT
│   ├── LR/f_001.tif        clinical CT (resliced to the micro-CT grid)
│   ├── SY/f_001.tif        synthetic low resolution
│   └── MS/f_001.tif        bone mask
├── f_002/
└── ...
```

### Visualisations

Clinical scans are usually not detailed enough to show bone microstructure, because scan time and radiation dose are limited. Trabecular bone, the inner part of the bone, is where the microstructure is most detailed and where SR is most needed. The figure below shows example slices from the three modalities, with zoomed-in patches.

<figure><img class="figure" src="https://github.com/user-attachments/assets/4a5e778e-2bc9-4299-880a-32f57b2e8280" alt="Example slices of clinical CT, micro-CT and synthetic data with zoomed-in patches"></figure>

<figure>
  <img class="figure" src="https://github.com/user-attachments/assets/d8f591f7-f5bc-4677-8526-9799362ccce4" alt="Animated centre slices of one bone in all three planes for all modalities">
  <figcaption>Centre slices of one bone in all three planes, for every modality.</figcaption>
</figure>

### Download

You can download the data from the [DTU data archive](https://archive.compute.dtu.dk/files/public/projects/FACTS). For a stable download that can resume if interrupted, copy the link of a file from the archive and use `wget -c`:

```bash
wget -c -O f_001.zip "https://archive.compute.dtu.dk/downloads/public/projects/FACTS/f_001.zip"
```

### Usage

```python
import tifffile
import matplotlib.pyplot as plt

root = "/path/to/FACTS"
vols = {mod: tifffile.imread(f"{root}/f_001/{mod}/f_001.tif") for mod in ["LR", "SY", "HR", "MS"]}

idx = 800
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, (mod, vol) in zip(axes, vols.items()):
    ax.imshow(vol[:, :, idx], cmap="gray")
    ax.set_title(mod)
    ax.axis("off")
plt.show()
```

## Preprocessing {#preprocessing}

The code for all steps is in the [GitHub repository](https://github.com/phialosophy10/FACTS).

### Intensity calibration

The clinical CT (in Hounsfield units) and the micro-CT have very different intensity scales (see the table above). To let a network learn the mapping from LR to HR, we bring the micro-CT to the scale of the clinical CT with a **linear calibration** per femur:

<p class="callout">HR<sub>cal</sub> = <i>a</i> &middot; HR + <i>b</i></p>

The clinical CT is essentially a blurred version of the same object on a linear intensity scale. So *a* and *b* are fitted by least squares such that *a* &middot; G<sub>&sigma;</sub>(HR) + *b* matches the clinical CT inside the bone mask, where G<sub>&sigma;</sub> is a Gaussian blur. The blur width &sigma; is chosen per femur by the best fit (R<sup>2</sup> = 0.70–0.95; &sigma; = 4 micro-CT voxels for all femurs except one). The calibrated volume is saved as `HR/f_NNN_hm.tif` (uint16, shifted by +2048 so that air fits in the unsigned range).

A linear mapping keeps the contrast and noise proportions of the micro-CT. We first tried histogram matching, which makes the two histograms identical. Because the micro-CT has sharp, separated intensity peaks while the clinical CT is blurred, that mapping is strongly non-linear: it stretched some intensity ranges, and their noise, and compressed others.

{% assign hm = site.data.results.histmatch %}
{% if hm.size > 0 %}
<figure>
  <img class="figure" src="{{ '/assets/img/' | append: hm[0].file | relative_url }}" alt="Intensity histograms before and after calibration">
  <figcaption>Intensity histograms inside the bone mask for {{ hm[0].femur }}. Left: original micro-CT. Right: clinical CT and calibrated micro-CT. The histograms do not need to coincide: thin trabeculae are blurred in the clinical CT, so its bone values are lower than in the sharp, calibrated micro-CT.</figcaption>
</figure>
{% endif %}

### Synthetic low-resolution data

To compare with the standard SR setting, we also make a synthetic low-resolution version of the calibrated micro-CT. Each 2D slice is

1. blurred with a Gaussian kernel (&sigma; = 1.2),
2. downsampled by a factor of 4, and
3. upsampled 4&times; with linear interpolation, back to the original grid.

The result is saved as `SY/f_NNN_hm.tif`. Because the synthetic data is made from the calibrated HR volume, the real input, the synthetic input and the target all share one intensity scale.

## Benchmark {#benchmark}

### Models

We train two 2D GAN architectures, adapted to inputs that are already on the high-resolution grid (the generators contain no upsampling layers):

* **SRGAN**[^srgan]: SRResNet generator with 16 residual blocks, trained with the standard GAN loss.
* **ESRGAN**[^esrgan]: RRDBNet generator with 8 residual-in-residual dense blocks, trained with the relativistic average GAN loss.

Both use the same VGG-style, fully convolutional discriminator. The generator loss is

<p class="callout"><i>L</i><sub>G</sub> = <i>L</i><sub>1</sub>(SR, HR) + 10<sup>-3</sup> &middot; <i>L</i><sub>adv</sub></p>

There is **no perceptual (VGG) loss**: in our earlier experiments it made no significant difference. Each generator is first pre-trained for 10k iterations with the L1 loss only. GAN training then continues to 100k iterations in total, using Adam (lr = 10<sup>-4</sup>, halved at 50% and 75% of training), batch size 16, and random 128 &times; 128 patches from inside the bone mask, with flips and rotations.

### Data split and experiments

* **Training:** f_001, f_013, f_015, f_021, f_026, f_074, f_075, f_083, f_164, f_172
* **Test:** f_002, f_086, f_138 (the same as in the MIUA 2024 paper)

Each model is trained on **real** input (clinical CT &rarr; micro-CT) and on **synthetic** input (synthetic LR &rarr; micro-CT). Each trained model is then tested on both real and synthetic input (cross-testing). Without SR, the input itself serves as the baseline.

### Metrics

All metrics are computed inside the bone mask, on every 10th slice of the test femurs. Intensities are normalised per femur from the micro-CT, in the same way for every method.

* **PSNR** and **SSIM:** voxel-wise similarity to the micro-CT.
* **LPIPS** (AlexNet)[^lpips]: learned perceptual distance. Lower is better.
* **BV/TV:** bone volume fraction, after Otsu thresholding inside the mask.
* **Tb.Th:** mean trabecular thickness, from the 2D local thickness of the binarised bone.

&Delta; denotes SR minus micro-CT, so 0 is ideal.

## Results {#results}

### Real (clinical CT) input

{% include results_table.html test_key="real" %}

### Synthetic input

{% include results_table.html test_key="synth" %}

Values are the mean &plusmn; standard deviation over the three test femurs.

{% assign figs = site.data.results.figures %}
{% if figs.size > 0 %}
### Visual comparison

{% for f in figs %}
<figure>
  <img class="figure" src="{{ '/assets/img/results/' | append: f.file | relative_url }}" alt="SR results for {{ f.femur }}, {{ f.test_key }} input">
  <figcaption>{{ f.femur }}, {% if f.test_key == "real" %}clinical CT{% else %}synthetic{% endif %} input. Top: full slice (bone bounding box). Bottom: zoom on the marked region.</figcaption>
</figure>
{% endfor %}
{% endif %}

{% assign pf = site.data.results.per_femur %}
{% if pf.size > 0 %}
<details>
<summary>Per-femur results</summary>
<div class="table-scroll"><table>
<thead><tr><th>Femur</th><th>Test input</th><th>Model</th><th>Trained on</th><th class="num">PSNR</th><th class="num">SSIM</th><th class="num">LPIPS</th><th class="num">BV/TV (HR)</th><th class="num">Tb.Th µm (HR)</th></tr></thead>
<tbody>
{% for r in pf %}
<tr><td>{{ r.femur }}</td><td>{{ r.test_data }}</td><td>{{ r.model }}</td><td>{{ r.train_data }}</td>
<td class="num">{{ r.psnr | round: 2 }}</td><td class="num">{{ r.ssim | round: 3 }}</td><td class="num">{% if r.lpips %}{{ r.lpips | round: 3 }}{% else %}-{% endif %}</td>
<td class="num">{{ r.bvtv | round: 3 }} ({{ r.bvtv_hr | round: 3 }})</td><td class="num">{{ r.tbth_um | round: 0 }} ({{ r.tbth_hr_um | round: 0 }})</td></tr>
{% endfor %}
</tbody></table></div>
</details>
{% endif %}

### Earlier results (MIUA 2024)

The figure below shows selected results from the paper. Left: ESRGAN trained and tested on real data. Centre: ESRGAN trained and tested on synthetic data. Right: SRGAN trained and tested on real data.

<figure><img class="figure" src="https://github.com/phialosophy10/BoneSuperResolution/assets/93533251/73cf9ea8-0ace-49cd-8df8-d157a88b458f" alt="Selected SR results from the MIUA 2024 paper"></figure>

## Citing {#citing}

If you use the dataset in your research, please cite the paper:

```bibtex
@inproceedings{bardenfleth2024superresolution,
  title     = {Superresolution of Real-World Multiscale Bone {CT} Verified with Clinical Bone Measures},
  author    = {Bardenfleth, Sophia W. and Dahl, Vedrana A. and Villa, Chiara and Kazakia, Galateia J. and Dahl, Anders B.},
  booktitle = {Medical Image Understanding and Analysis (MIUA)},
  series    = {Lecture Notes in Computer Science},
  pages     = {160--173},
  year      = {2024},
  publisher = {Springer},
  doi       = {10.1007/978-3-031-66958-3_12}
}
```

## References

[^srgan]: C. Ledig et al., *Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network*, CVPR 2017. [doi:10.1109/CVPR.2017.19](https://doi.org/10.1109/CVPR.2017.19)
[^esrgan]: X. Wang et al., *ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks*, ECCV Workshops 2018. [doi:10.1007/978-3-030-11021-5_5](https://doi.org/10.1007/978-3-030-11021-5_5)
[^lpips]: R. Zhang et al., *The Unreasonable Effectiveness of Deep Features as a Perceptual Metric*, CVPR 2018.
