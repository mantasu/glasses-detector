import sys

from PIL import Image

sys.path.append("../src")

from glasses_detector import GlassesDetector, GlassesSegmenter

SAMPLES = {
    "classification": {
        "eyeglasses": "data/classification/eyeglasses/sunglasses-glasses-detect/test/eyeglasses/face-149_png.rf.cc420d484a00dd7158550510785f0d51.jpg",
        "sunglasses": "data/classification/sunglasses/face-attributes-grouped/test/sunglasses/2014-Colored-Mirror-Sunglasses-2014-Renkli-ve-Aynal-Caml-Gunes-Gozlugu-Modelleri-19.jpg",
        "no-glasses": "data/classification/eyeglasses/face-attributes-extra/test/no_eyeglasses/800px_COLOURBOX4590613.jpg",
    },
    "segmentation-frames": [
        {
            "img": "data/segmentation/frames/eyeglass/val/images/Woman-wearing-white-rimmed-thin-spectacles_jpg.rf.097bc1a8d872b6acddc3312e95a2d48e.jpg",
            "msk": "data/segmentation/frames/eyeglass/val/masks/Woman-wearing-white-rimmed-thin-spectacles_jpg.rf.097bc1a8d872b6acddc3312e95a2d48e.jpg",
        },
        {
            "img": "data/segmentation/frames/glasses-segmentation-synthetic/test/images/img-Glass001-386-2_mouth_open-3-missile_launch_facility_01-037.jpg",
            "msk": "data/segmentation/frames/glasses-segmentation-synthetic/test/masks/img-Glass001-386-2_mouth_open-3-missile_launch_facility_01-037.jpg",
        },
        {
            "img": "data/segmentation/frames/glasses-segmentation-synthetic/test/images/img-Glass021-450-4_brow_lower-2-entrance_hall-204.jpg",
            "msk": "data/segmentation/frames/glasses-segmentation-synthetic/test/masks/img-Glass021-450-4_brow_lower-2-entrance_hall-204.jpg",
        },
    ],
    "segmentation-full": [
        {
            "img": "data/segmentation/full/celeba-mask-hq/test/images/821.jpg",
            "msk": "data/segmentation/full/celeba-mask-hq/test/masks/821.jpg",
        },
        {
            "img": "data/segmentation/full/celeba-mask-hq/test/images/1034.jpg",
            "msk": "data/segmentation/full/celeba-mask-hq/test/masks/1034.jpg",
        },
        {
            "img": "data/segmentation/full/celeba-mask-hq/test/images/2442.jpg",
            "msk": "data/segmentation/full/celeba-mask-hq/test/masks/2442.jpg",
        },
    ],
    "segmentation-legs": [
        {
            "img": "data/segmentation/legs/capstone-mini-2/test/images/IMG20230325193452_0_jpg.rf.ea87e7fe943f39216cacc84b32848e28.jpg",
            "msk": "data/segmentation/legs/capstone-mini-2/test/masks/IMG20230325193452_0_jpg.rf.ea87e7fe943f39216cacc84b32848e28.jpg",
        },
        {
            "img": "data/segmentation/legs/sunglasses-color-detection/test/images/2004970PJPXT_P00_JPG_jpg.rf.e1cd193efd84dac31c027e3d3649ec7a.jpg",
            "msk": "data/segmentation/legs/sunglasses-color-detection/test/masks/2004970PJPXT_P00_JPG_jpg.rf.e1cd193efd84dac31c027e3d3649ec7a.jpg",
        },
        {
            "img": "data/segmentation/legs/sunglasses-color-detection/test/images/aug_57_203675009QQT_P00_JPG_jpg.rf.54bdfef21f854be18d9dcf13fa5a7ae7.jpg",
            "msk": "data/segmentation/legs/sunglasses-color-detection/test/masks/aug_57_203675009QQT_P00_JPG_jpg.rf.54bdfef21f854be18d9dcf13fa5a7ae7.jpg",
        },
    ],
    "segmentation-lenses": [
        {
            "img": "data/segmentation/lenses/glasses-lens/test/images/face-35_jpg.rf.f0a9a1d3b4f9e756488294d2db1720d5.jpg",
            "msk": "data/segmentation/lenses/glasses-lens/test/masks/face-35_jpg.rf.f0a9a1d3b4f9e756488294d2db1720d5.jpg",
        },
        {
            "img": "data/segmentation/lenses/glass-color/test/images/2025260PJPVP_P00_JPG_jpg.rf.aaa9e83edbfd8a3c107650b62ddf52ed.jpg",
            "msk": "data/segmentation/lenses/glass-color/test/masks/2025260PJPVP_P00_JPG_jpg.rf.aaa9e83edbfd8a3c107650b62ddf52ed.jpg",
        },
        {
            "img": "data/segmentation/lenses/glasses-segmentation-cropped-faces/test/images/face-1306_scaled_cropping_jpg.rf.b5f5b788fb75aa05a15e69b938704c12.jpg",
            "msk": "data/segmentation/lenses/glasses-segmentation-cropped-faces/test/masks/face-1306_scaled_cropping_jpg.rf.b5f5b788fb75aa05a15e69b938704c12.jpg",
        },
    ],
    "segmentation-shadows": [
        {
            "img": "data/segmentation/shadows/glasses-segmentation-synthetic/test/images/img-Glass021-435-16_sadness-2-versveldpas-105.jpg",
            "msk": "data/segmentation/shadows/glasses-segmentation-synthetic/test/masks/img-Glass021-435-16_sadness-2-versveldpas-105.jpg",
        },
        {
            "img": "data/segmentation/shadows/glasses-segmentation-synthetic/test/images/img-Glass001-379-4_brow_lower-1-simons_town_rocks-315.jpg",
            "msk": "data/segmentation/shadows/glasses-segmentation-synthetic/test/masks/img-Glass001-379-4_brow_lower-1-simons_town_rocks-315.jpg",
        },
        {
            "img": "data/segmentation/shadows/glasses-segmentation-synthetic/test/images/img-Glass018-422-7_jaw_left-1-urban_street_03-039.jpg",
            "msk": "data/segmentation/shadows/glasses-segmentation-synthetic/test/masks/img-Glass018-422-7_jaw_left-1-urban_street_03-039.jpg",
        },
    ],
    "segmentation-smart": [
        {
            "img": "data/segmentation/smart/face-synthetics-glasses/test/images/000410.jpg",
            "msk": "data/segmentation/smart/face-synthetics-glasses/test/masks/000410.jpg",
        },
        {
            "img": "data/segmentation/smart/face-synthetics-glasses/test/images/001229.jpg",
            "msk": "data/segmentation/smart/face-synthetics-glasses/test/masks/001229.jpg",
        },
        {
            "img": "data/segmentation/smart/face-synthetics-glasses/test/images/002315.jpg",
            "msk": "data/segmentation/smart/face-synthetics-glasses/test/masks/002315.jpg",
        },
    ],
    "detection-eyes": [
        {
            "img": "data/detection/eyes/ex07/test/images/face-16_jpg.rf.9554ce9ff29cca368918cb849806902f.jpg",
            "ann": "data/detection/eyes/ex07/test/annotations/face-16_jpg.rf.9554ce9ff29cca368918cb849806902f.txt",
        },
        {
            "img": "data/detection/eyes/glasses-detection/test/images/41d3e9440d1678109133_jpeg.rf.564dc61348a3986faf801d352a7ebe41.jpg",
            "ann": "data/detection/eyes/glasses-detection/test/annotations/41d3e9440d1678109133_jpeg.rf.564dc61348a3986faf801d352a7ebe41.txt",
        },
        {
            "img": "data/detection/eyes/glasses-detection/test/images/woman-face-eyes-feeling_jpg.rf.8c1547d76fe23936984db74a5507f188.jpg",
            "ann": "data/detection/eyes/glasses-detection/test/annotations/woman-face-eyes-feeling_jpg.rf.8c1547d76fe23936984db74a5507f188.txt",
        },
    ],
    "detection-solo": [
        {
            "img": "data/detection/solo/onlyglasses/test/images/8--52-_jpg.rf.cfc2d6dec8f46cd5b91c9c112fbb8bf3.jpg",
            "ann": "data/detection/solo/onlyglasses/test/annotations/8--52-_jpg.rf.cfc2d6dec8f46cd5b91c9c112fbb8bf3.txt",
        },
        {
            "img": "data/detection/solo/kacamata-membaca/test/images/85_jpg.rf.4c164fa95a20bebc7c888d34ed160e16.jpg",
            "ann": "data/detection/solo/kacamata-membaca/test/annotations/85_jpg.rf.4c164fa95a20bebc7c888d34ed160e16.txt",
        },
        {
            "img": "data/detection/worn/ai-pass/test/images/28f3d11c3465ce2e74d8a4d65861de51_jpg.rf.e119438402f655cec8032304c7603606.jpg",
            "ann": "data/detection/worn/ai-pass/test/annotations/28f3d11c3465ce2e74d8a4d65861de51_jpg.rf.e119438402f655cec8032304c7603606.txt",
        },
    ],
    "detection-worn": [
        {
            "img": "data/detection/worn/glasses-detection/test/images/425px-robert_downey_jr_avp_iron_man_3_paris_jpg.rf.998f29000b52081eb6ea4d25df75512c.jpg",
            "ann": "data/detection/worn/glasses-detection/test/annotations/425px-robert_downey_jr_avp_iron_man_3_paris_jpg.rf.998f29000b52081eb6ea4d25df75512c.txt",
        },
        {
            "img": "data/detection/worn/ai-pass/test/images/glasses120_png_jpg.rf.847610bd1230c85c8f81cbced18c38ea.jpg",
            "ann": "data/detection/worn/ai-pass/test/annotations/glasses120_png_jpg.rf.847610bd1230c85c8f81cbced18c38ea.txt",
        },
        {
            "img": "data/detection/worn/ai-pass/test/images/women-with-glass_81_jpg.rf.bef8096d89dd0805ff3bbf0f8d08b0c8.jpg",
            "ann": "data/detection/worn/ai-pass/test/annotations/women-with-glass_81_jpg.rf.bef8096d89dd0805ff3bbf0f8d08b0c8.txt",
        },
    ],
}


def generate_examples(data_dir: str = "..", out_dir: str = "_static/img"):
    for task, samples in SAMPLES.items():
        if task == "classification":
            for label, path in samples.items():
                # Load the image and save it
                img = Image.open(f"{data_dir}/{path}")
                img.save(f"{out_dir}/{task}-{label}.jpg")
        elif task.startswith("detection"):
            for i, sample in enumerate(samples):
                # Load the image
                img = Image.open(f"{data_dir}/{sample["img"]}")

                with open(f"{data_dir}/{sample["ann"]}", "r") as f:
                    # Load annotations (single bbox per image)
                    ann = [list(map(float, f.read().split()))]

                # Draw the bounding box and save the image
                out = GlassesDetector.draw_boxes(img, ann, colors="red", width=3)
                out.save(f"{out_dir}/{task}-{i}.jpg")
        elif task.startswith("segmentation"):
            for i, sample in enumerate(samples):
                # Load image and mask and overlay them
                img = Image.open(f"{data_dir}/{sample["img"]}")
                msk = Image.open(f"{data_dir}/{sample["msk"]}")
                out = GlassesSegmenter.draw_masks(img, msk, colors="red", alpha=0.5)
                out.save(f"{out_dir}/{task}-{i}.jpg")


if __name__ == "__main__":
    # cd docs/
    # python helpers/generate_examples.py
    generate_examples()
