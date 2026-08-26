import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

export default function RunPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold text-white">Run the continuation on your xBD subset</h1>
        <p className="mt-2 max-w-2xl text-sm text-slate-400">
          Keep the crops you already extracted. Point the scripts at that folder instead of
          the synthetic demo tiles shipped with this repo.
        </p>
      </div>

      <Card className="bg-[#0d1724] ring-white/10">
        <CardHeader>
          <CardTitle>Expected folders</CardTitle>
          <CardDescription>
            Matches a typical xBD download plus the building-extraction notebook you already ran.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <pre className="overflow-x-auto rounded-lg bg-black/40 p-4 text-xs text-slate-300">{`data/
  raw/
    images/   *_pre_disaster.png  *_post_disaster.png
    labels/   matching JSON with features.xy polygons
  crops/      output of extract_buildings.py
    no-damage/
    minor-damage/
    major-damage/
    destroyed/
    manifest.csv
  splits/     train.csv val.csv test.csv`}</pre>
        </CardContent>
      </Card>

      <div className="grid gap-4 md:grid-cols-2">
        <Step
          n="1"
          title="Re-extract if you only saved tiles"
          body="If you already have crops, skip this. Otherwise:"
          code={`cd ml
python extract_buildings.py \\
  --images ../data/raw/images \\
  --labels ../data/raw/labels \\
  --out ../data/crops --size 224`}
        />
        <Step
          n="2"
          title="Split without leaking disasters"
          body="Prefers holding out whole events so test tiles are unseen."
          code={`python prepare_splits.py \\
  --manifest ../data/crops/manifest.csv \\
  --out ../data/splits`}
        />
        <Step
          n="3"
          title="Fine-tune ResNet50"
          body="Freeze the backbone on CPU. Unfreeze on Colab GPU after the head converges."
          code={`python train.py --root .. --epochs 8 \\
  --freeze-backbone --batch-size 16
# GPU:
python train.py --root .. --epochs 20 \\
  --unfreeze-backbone --lr 3e-5`}
        />
        <Step
          n="4"
          title="Evaluate and map severity"
          body="Writes JSON the dashboard can load if you copy it into web/public/demo/."
          code={`python evaluate.py --root .. \\
  --checkpoint ../models/resnet50_best.pt
python severity.py \\
  --predictions ../web/public/demo/metrics.json`}
        />
      </div>

      <Card className="bg-[#0d1724] ring-white/10">
        <CardHeader>
          <CardTitle>Colab</CardTitle>
          <CardDescription>
            Open notebooks/02_resnet50_damage_classification.ipynb. Mount Drive, set ROOT to
            the folder where you saved crops, and run the training cells with a GPU runtime.
          </CardDescription>
        </CardHeader>
        <CardContent className="text-sm text-slate-400">
          Do not chase accuracy. Report weighted F1 and the minor/major confusion pair. Those
          two numbers tell a reviewer you understood xBD.
        </CardContent>
      </Card>
    </div>
  );
}

function Step({
  n,
  title,
  body,
  code,
}: {
  n: string;
  title: string;
  body: string;
  code: string;
}) {
  return (
    <Card className="bg-[#0d1724] ring-white/10">
      <CardHeader>
        <CardDescription>Step {n}</CardDescription>
        <CardTitle>{title}</CardTitle>
        <CardDescription>{body}</CardDescription>
      </CardHeader>
      <CardContent>
        <pre className="overflow-x-auto rounded-lg bg-black/40 p-3 text-xs text-slate-300">
          {code}
        </pre>
      </CardContent>
    </Card>
  );
}
