#!/usr/bin/env bash
# Download the extra v2 training sources from ModelScope at pinned per-file
# revisions, verify sha256, and convert them to JevForge records.
#
#   bash examples/classify/jev/data/fetch_v2_sources.sh [DATA_DIR]
#
# DATA_DIR (default ~/data) receives:
#   open-jev-v1/        ZefanCai/Open-Jev, all 12 configs (~49 MB, CC0)
#   jevembed-data/      HIT-TMG/JevEmbed-Data (~370 MB, per-row licenses)
#   jev-records/open-jev-v1, jev-records/jevembed   converted records
# JevEmbed sources are whitelisted in convert_datasets.py (JEVEMBED_SOURCES).
set -euo pipefail

DATA_DIR="${1:-$HOME/data}"
HERE="$(cd "$(dirname "$0")" && pwd)"
OPEN_JEV_V1="https://www.modelscope.cn/api/v1/datasets/ZefanCai/Open-Jev/repo"
JEVEMBED="https://www.modelscope.cn/api/v1/datasets/HIT-TMG/JevEmbed-Data/repo"

# dataset path revision sha256
FILES="
open-jev-v1 data/ir-control-v1/calibration-00000-of-00001.parquet 2cef672910ef5d810b1eb920256f02a8ae56f453 864be79af98ec1145637b42882427f9f31e1de55628255b58eb19e6809563217
open-jev-v1 data/release-v2-redistributable/calibration-00000-of-00001.parquet 27375083f76afce844e778371d26be3acaf2fe3c 74cf0064f68dcccbcedd4849d3b347f4c900bdd5329a78ec249fcdf24f89b554
open-jev-v1 data/browser-drone-expansion-v1-redistributable/calibration-00000-of-00001.parquet 2cef672910ef5d810b1eb920256f02a8ae56f453 20e24d12d308841b5d3249e22a16832625436e76861bf39742d436bcb4e1c557
open-jev-v1 data/mailroom-control-v1/calibration-00000-of-00001.parquet 2cef672910ef5d810b1eb920256f02a8ae56f453 ad584458b9d498702d2c3fd5cb4427a8ea57e588ad3cc31a3cca3609213ccc3b
open-jev-v1 data/sponsor-segment-control-v1/calibration-00000-of-00001.parquet 27375083f76afce844e778371d26be3acaf2fe3c 8d6e546d522ac9c4520a11c790fed61e13d0bfb986dae0711d2ceed696f7a85a
open-jev-v1 data/amount-extraction-control-v1/calibration-00000-of-00001.parquet 2cef672910ef5d810b1eb920256f02a8ae56f453 73fa34f9a7c9a951efb542bfb91d360b40b8f926c121fef7b3dbe75e65d10937
open-jev-v1 data/entity-alignment-control-v1/calibration-00000-of-00001.parquet 2cef672910ef5d810b1eb920256f02a8ae56f453 5264036ff383944aaa608129c5edfd0637dc8ad1bbeb1f0045f88e95c973d395
open-jev-v1 data/context-retention-control-v1/calibration-00000-of-00001.parquet 2cef672910ef5d810b1eb920256f02a8ae56f453 852a87c4816242f656ace8f7ab31c7d39e3643da452ec6bb0f3d40d1da8bbf3a
open-jev-v1 data/phone-extraction-control-v1/calibration-00000-of-00001.parquet 27375083f76afce844e778371d26be3acaf2fe3c d9ba25e4d8c99d2b4f6a79f29a47735318105f7e8e8e571e255021a0626e4b87
open-jev-v1 data/silent-failure-control-v1/calibration-00000-of-00001.parquet 27375083f76afce844e778371d26be3acaf2fe3c 5a8d4c1612139064a00202a8f1f2ba92389dd8c4f42d9a5521edf8895516f01b
open-jev-v1 data/citation-control-v1/calibration-00000-of-00001.parquet 2cef672910ef5d810b1eb920256f02a8ae56f453 4f420cc1b004f6ba2024e96d5c64451a3434acd5759be8d0718e28b9fcd10ed2
open-jev-v1 data/email-selection-control-v1/calibration-00000-of-00001.parquet 2cef672910ef5d810b1eb920256f02a8ae56f453 2fca99e888109d3a4e223e6b38f562c1c0f2563d62a48b831d778d10d10b8ac3
open-jev-v1 data/release-v2-redistributable/ood-00000-of-00001.parquet 27375083f76afce844e778371d26be3acaf2fe3c ba8cb3f95b9121aa73e4b997ac9c350e2ec4dddb972cb5c3086da911f0deb9ef
open-jev-v1 data/browser-drone-expansion-v1-redistributable/ood-00000-of-00001.parquet 2cef672910ef5d810b1eb920256f02a8ae56f453 6f03b875a21e8a4c1868c8334455fe78d97b6a6a74f8a12b87374fa1d01f6d71
open-jev-v1 data/email-selection-control-v1/ood-00000-of-00001.parquet 2cef672910ef5d810b1eb920256f02a8ae56f453 bb74db37451174ce8684eeddf0e846b80d65e658e2b74c3697a77aa8d42e948b
open-jev-v1 data/amount-extraction-control-v1/ood-00000-of-00001.parquet 2cef672910ef5d810b1eb920256f02a8ae56f453 2ea2b28dc09d2048644f321cfecaaea149f025d28fdc5d91beef1ec76200f3a3
open-jev-v1 data/phone-extraction-control-v1/ood-00000-of-00001.parquet 27375083f76afce844e778371d26be3acaf2fe3c a344d08e4842b2a9d1372ef6b34851be332aeaa9ee657acebe0b183480fb67ad
open-jev-v1 data/context-retention-control-v1/ood-00000-of-00001.parquet 2cef672910ef5d810b1eb920256f02a8ae56f453 50f3eae83f6b02f5c53ad80693bff2c1959255449567e671ab93da30319750a5
open-jev-v1 data/mailroom-control-v1/ood-00000-of-00001.parquet 2cef672910ef5d810b1eb920256f02a8ae56f453 4d14896ef69f5ac52899d93cd019f80409373622f7707346c4c9fac9814a3964
open-jev-v1 data/silent-failure-control-v1/ood-00000-of-00001.parquet 27375083f76afce844e778371d26be3acaf2fe3c bfa8faf6abad9a54fe3507bbeeaea1a087ea0e5b0c4169bfeef58a15129f742d
open-jev-v1 data/entity-alignment-control-v1/ood-00000-of-00001.parquet 2cef672910ef5d810b1eb920256f02a8ae56f453 a6ff9fcc05517788a40554712e1a1e77e25a76c53fa8d712b3b17324c74b6ced
open-jev-v1 data/sponsor-segment-control-v1/ood-00000-of-00001.parquet 27375083f76afce844e778371d26be3acaf2fe3c e1d2cfc5b5a6404999dee9f5fe3e10e59091854688c61ba93a701e9a103532fb
open-jev-v1 data/ir-control-v1/ood-00000-of-00001.parquet 2cef672910ef5d810b1eb920256f02a8ae56f453 52062c0f76afe608df91d780be739c67330cd7f4b538470216f0b3fe662fe1da
open-jev-v1 data/citation-control-v1/ood-00000-of-00001.parquet 2cef672910ef5d810b1eb920256f02a8ae56f453 59c526a897b449d45483d7c26c5e8e55e609175e7e6c3b5e2375346da420c183
open-jev-v1 data/email-selection-control-v1/test-00000-of-00001.parquet 2cef672910ef5d810b1eb920256f02a8ae56f453 0b2aff2ca90e68d8346b4ea5cc23b045aad3f786a27371f8d9edb55feb677e02
open-jev-v1 data/context-retention-control-v1/test-00000-of-00001.parquet 2cef672910ef5d810b1eb920256f02a8ae56f453 6c65123feabbabe52abf5573e6345a7922d42c5c68cdfd98bd7d3476f212434c
open-jev-v1 data/sponsor-segment-control-v1/test-00000-of-00001.parquet 27375083f76afce844e778371d26be3acaf2fe3c 81c923e07cb3c86b0ccac7f56fe8236454eac544f3f9c8e8f600559d6f1bae52
open-jev-v1 data/ir-control-v1/test-00000-of-00001.parquet 2cef672910ef5d810b1eb920256f02a8ae56f453 082495de2ca3c8487349c06df8f3bc6bcf6ff8128ec729616fd1eed146ee2c56
open-jev-v1 data/citation-control-v1/test-00000-of-00001.parquet 2cef672910ef5d810b1eb920256f02a8ae56f453 e77adf6898d73beed303edd6bcceb9e1cd6fb6a01b9055c3804dc46acd225c91
open-jev-v1 data/silent-failure-control-v1/test-00000-of-00001.parquet 27375083f76afce844e778371d26be3acaf2fe3c d1e82f2183c6ae7d7858552381349e62575c526407ed9c2acadedf5bc99e7cca
open-jev-v1 data/release-v2-redistributable/test-00000-of-00001.parquet 27375083f76afce844e778371d26be3acaf2fe3c 05edb50da60abb087328e353b201d83b745e01393b4a43b5320b90720a8132ff
open-jev-v1 data/entity-alignment-control-v1/test-00000-of-00001.parquet 2cef672910ef5d810b1eb920256f02a8ae56f453 76dd1037bb61ccf4af34f6018432479573a2f95c48136afea50e8e886e5f36ad
open-jev-v1 data/browser-drone-expansion-v1-redistributable/test-00000-of-00001.parquet 2cef672910ef5d810b1eb920256f02a8ae56f453 afc858fc1724a085375333d14bae5e4750fea4ab934f316e9b9b4c9890921a89
open-jev-v1 data/mailroom-control-v1/test-00000-of-00001.parquet 2cef672910ef5d810b1eb920256f02a8ae56f453 605a9f4da39da089f8f9020e57ec57999ed5451ff3506c5629dbd056d5594280
open-jev-v1 data/amount-extraction-control-v1/test-00000-of-00001.parquet 2cef672910ef5d810b1eb920256f02a8ae56f453 deaa97a33988176b7c3022ab02b22b678ec8fd10f95d2010573097858fc9491f
open-jev-v1 data/phone-extraction-control-v1/test-00000-of-00001.parquet 27375083f76afce844e778371d26be3acaf2fe3c 5e105d471975be41e2680243d5362d4447350c843904cb2aa1536b5675a66066
open-jev-v1 data/context-retention-control-v1/train-00000-of-00001.parquet 2cef672910ef5d810b1eb920256f02a8ae56f453 ad416a9cb0bd7692ef978c799458edb0fa6c4fbc7b545355c3ccd3a52e420180
open-jev-v1 data/amount-extraction-control-v1/train-00000-of-00001.parquet 2cef672910ef5d810b1eb920256f02a8ae56f453 dd09738e9463fd1efcdf87137600bea0c21270ee729c4475124d684b88d0cbb2
open-jev-v1 data/email-selection-control-v1/train-00000-of-00001.parquet 2cef672910ef5d810b1eb920256f02a8ae56f453 9c91b6293515a21e709dd8fe5a05a05374e20933f0f8f6c4c847ec96707188c0
open-jev-v1 data/sponsor-segment-control-v1/train-00000-of-00001.parquet 27375083f76afce844e778371d26be3acaf2fe3c a99d5415363b9212ce3c9017a6f8345c88888217e10aa2d51115c090639a349a
open-jev-v1 data/phone-extraction-control-v1/train-00000-of-00001.parquet 27375083f76afce844e778371d26be3acaf2fe3c b4a4017adb54678881cacccd9d5b08260a2027f79934daaa9b00831a6719e3a3
open-jev-v1 data/entity-alignment-control-v1/train-00000-of-00001.parquet 2cef672910ef5d810b1eb920256f02a8ae56f453 6a07f77897960685947c07736ec7f7a6e2977ec733accf17d6b1707d88360a4b
open-jev-v1 data/ir-control-v1/train-00000-of-00001.parquet 2cef672910ef5d810b1eb920256f02a8ae56f453 e0788eba891c706dc2fd8f92d95f7172d0f5a482ab7d899d4cde312d23b9b9a5
open-jev-v1 data/browser-drone-expansion-v1-redistributable/train-00000-of-00001.parquet 2cef672910ef5d810b1eb920256f02a8ae56f453 a4fdfdff0d9f603d3e7ba51d5a48b2cd0d8eeb60f3244779b6b04d135f4a89f7
open-jev-v1 data/citation-control-v1/train-00000-of-00001.parquet 2cef672910ef5d810b1eb920256f02a8ae56f453 8402d2668c4c46bf9f5c3a8bc292652c8de72d288e8e5c0df30fd13d73b5c51d
open-jev-v1 data/release-v2-redistributable/train-00000-of-00001.parquet 27375083f76afce844e778371d26be3acaf2fe3c a76be013b2986a97e4d61faad4fd5f0a6ddb934760d5dcf3161aaa118909b6bf
open-jev-v1 data/silent-failure-control-v1/train-00000-of-00001.parquet 27375083f76afce844e778371d26be3acaf2fe3c 0aadeb3372275a678226bfa8371bca6a5e6260cff6aad20216103271bf9b8721
open-jev-v1 data/mailroom-control-v1/train-00000-of-00001.parquet 27375083f76afce844e778371d26be3acaf2fe3c 5bc38433ab3758fc31d56a2f4f9e1aebb58881d1f21edf1bffdce49f38d43d32
open-jev-v1 data/email-selection-control-v1/validation-00000-of-00001.parquet 2cef672910ef5d810b1eb920256f02a8ae56f453 17d48790b255add53d5d0894078e11cc72d8a6ee60008c06144dc61f53789173
open-jev-v1 data/entity-alignment-control-v1/validation-00000-of-00001.parquet 2cef672910ef5d810b1eb920256f02a8ae56f453 e08ac592db16a3c12c078f1b176b3f1a19170b6f9615f1d93e56fca971bf4766
open-jev-v1 data/browser-drone-expansion-v1-redistributable/validation-00000-of-00001.parquet 2cef672910ef5d810b1eb920256f02a8ae56f453 a70ba9b5700c128e529baf5cca6224743820d5501fe246b6d898851651f141ef
open-jev-v1 data/ir-control-v1/validation-00000-of-00001.parquet 2cef672910ef5d810b1eb920256f02a8ae56f453 4b95b2421e6cb6db45cac0bff10a58e8770789e6a973cf280fde99d25a81f7f7
open-jev-v1 data/amount-extraction-control-v1/validation-00000-of-00001.parquet 2cef672910ef5d810b1eb920256f02a8ae56f453 b555514e972eb0f38ac6e01691068aeb2f02b26e9e67754b13b908c80c4adc25
open-jev-v1 data/silent-failure-control-v1/validation-00000-of-00001.parquet 27375083f76afce844e778371d26be3acaf2fe3c 8d04d313314248106893e2578b80537c34e1b6acf10c2cdabe8802bbd94f6698
open-jev-v1 data/citation-control-v1/validation-00000-of-00001.parquet 2cef672910ef5d810b1eb920256f02a8ae56f453 faa557577101f587d6bd623136c32181c0474077e34c1168a400d458a1719f80
open-jev-v1 data/context-retention-control-v1/validation-00000-of-00001.parquet 2cef672910ef5d810b1eb920256f02a8ae56f453 7e5fb2ce3d231856a1cd1fa74f0b8ced4b76cec9ec6d0b43f7450047ab62ef57
open-jev-v1 data/release-v2-redistributable/validation-00000-of-00001.parquet 27375083f76afce844e778371d26be3acaf2fe3c 5563764b2d92306e2bfdfd883edfee28a82acf36699f6533c0bd51978b784e59
open-jev-v1 data/phone-extraction-control-v1/validation-00000-of-00001.parquet 27375083f76afce844e778371d26be3acaf2fe3c 7dfaa40f36f85e170f6c323cbbc26651bd88c52d8bcb1bd1a5f2c7ad0e82a035
open-jev-v1 data/mailroom-control-v1/validation-00000-of-00001.parquet 27375083f76afce844e778371d26be3acaf2fe3c c60276b58d5856173b316266c5323c6683049ccca70b479210871371632ca317
open-jev-v1 data/sponsor-segment-control-v1/validation-00000-of-00001.parquet 27375083f76afce844e778371d26be3acaf2fe3c aab6645a8031fcf1be7e2ef97c603705bf96977025eddb215815ae55e0a4401c
jevembed data/test-00000-of-00001.parquet 673e8ce0f5fb6f0263832dad2721547745656b88 b8c2660c9b1d7e1d298ec943ecc273337c46a724e10b8e7700712da3dd9f91c4
jevembed data/train-00000-of-00009.parquet 6378687d8bec7828f4f4908d4b83a9e3b4d2559d f875388b1d699643869c97aa444452c53b09d53821039761da4bdf68377b5a97
jevembed data/train-00001-of-00009.parquet 9be65d458bebef2773b553ae8fe7e9f6d3fe00c4 ac3c58f01426d10b3b71d11df9ab11e58f539fc121fb856286074a162098f1ad
jevembed data/train-00002-of-00009.parquet d0020f8796a61cd0e86bdb710b1612451a01f1a3 8cd78225ee905e02c87d005612487c661b86c978c6f4e07ed06f40bc9921b6e8
jevembed data/train-00003-of-00009.parquet 4c62ff2ac88074b0ae04c0b7e2f51898460c3c57 01835c3193cf9028bb216d5b0477ff683726fec3b31893dd2b81d658e308258b
jevembed data/train-00004-of-00009.parquet efc22e18a38ca8b2b3470d58ac704851423e0841 cf92e5893ba92d224fa74ff8feec7037874fedfde8a7c49ecd23349d1e90d45f
jevembed data/train-00005-of-00009.parquet b62d2aba3d2bf169e12a126cc2a035b0b95331b1 6b911d9601fe026e77422930a9fb34548a941a2d0367a77dea91ed604cc1cb64
jevembed data/train-00006-of-00009.parquet 6f7f32fbccefe6688ff0c4a9c52c86d71a813e7f 215651e4390a371167a75f5e9aa1bc77483cf54ab2677f0cca218a3c54d0cf62
jevembed data/train-00007-of-00009.parquet 016af39a826ebff98a846e0cd341a7957df9eede 63f538efaf609317b603146e1296031c23fe0d095ccc276637f91c8a2f3f4e7c
jevembed data/train-00008-of-00009.parquet 4a5b8eab39810e7c76553ec53ee655c87485dc88 39a91452a827ac591f348e411cb7f850691b392781b09abc755e3436877091d7
"

while read -r dataset path revision sha; do
  [[ -z "$dataset" ]] && continue
  if [[ "$dataset" == "open-jev-v1" ]]; then
    file="$DATA_DIR/open-jev-v1/$path"; repo="$OPEN_JEV_V1"
  else
    file="$DATA_DIR/jevembed-data/$(basename "$path")"; repo="$JEVEMBED"
  fi
  mkdir -p "$(dirname "$file")"
  if [[ ! -s "$file" ]] || ! echo "$sha  $file" | sha256sum -c --status; then
    echo "download $dataset $path"
    curl -sfL --retry 3 -o "$file" "$repo?Revision=$revision&FilePath=$path"
  fi
  echo "$sha  $file" | sha256sum -c --quiet || { echo "sha256 mismatch for $file" >&2; exit 1; }
done <<< "$FILES"
for doc in README.md LICENSE-DATA.md; do
  [[ -s "$DATA_DIR/open-jev-v1/$doc" ]] || curl -sfL -o "$DATA_DIR/open-jev-v1/$doc" "$OPEN_JEV_V1?Revision=master&FilePath=$doc" || true
done
[[ -s "$DATA_DIR/jevembed-data/README.md" ]] || curl -sfL -o "$DATA_DIR/jevembed-data/README.md" "$JEVEMBED?Revision=master&FilePath=README.md" || true
echo "parquet verified under $DATA_DIR"

for name in open-jev-v1 jevembed; do
  records="$DATA_DIR/jev-records/$name"
  src="$DATA_DIR/$([[ $name == jevembed ]] && echo jevembed-data || echo open-jev-v1)"
  [[ -s "$records/manifest.json" ]] || python "$HERE/../convert_datasets.py" "$name" --src "$src" --out "$records"
done
echo "records under $DATA_DIR/jev-records"
