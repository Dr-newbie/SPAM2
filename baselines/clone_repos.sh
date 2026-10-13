#!/usr/bin/env bash
# Clone the upstream baseline implementations at the EXACT commits used for the
# results reported in the SPAM manuscript. Run once before using the harness:
#   bash baselines/clone_repos.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p repos && cd repos

clone_pin () {  # <url> <dir> <commit>
  if [ ! -d "$2/.git" ]; then git clone "$1" "$2"; fi
  git -C "$2" fetch --quiet origin || true
  git -C "$2" checkout --quiet "$3"
  echo "pinned  $2 @ $(git -C "$2" rev-parse --short=12 HEAD)"
}

clone_pin https://github.com/bryanhe/ST-Net.git                     ST-Net              43022c1cb7de1540d5a74ea2338a12c82491c5ad
clone_pin https://github.com/tmonjo/DeepSpaCE.git                   DeepSpaCE           5e023eaa4df1344046693c945cca7bd1099ef35a
clone_pin https://github.com/bowang-lab/BLEEP.git                   BLEEP               23959670c7407510c1716a173184bdcc18833a35
clone_pin https://github.com/wenwenmin/STMCL.git                    STMCL               60843b2d1a8a0cd7d29ccc35aada2bb2c942d400
clone_pin https://github.com/Yan98/EGN.git                          EGN                 bae8f0574d98124d723af259116c4375ae037b24
clone_pin https://github.com/PangeaResearch/enlight-deeppt-data.git enlight-deeppt-data d96708890937d595b513006fe1ff1e777e271be2

# DeepPT: the training code is distributed by the authors as an archive rather
# than a git repository (see Hoang et al., Nature Cancer 2024, code availability
# and the ENLIGHT-DeepPT repository above). Our harness re-implements the
# three-stage DeepPT pipeline (frozen ResNet50 features -> autoencoder -> MLP)
# in spam_bench/, and optionally loads the authors' released ResNet50 weights:
mkdir -p DeepPT
echo "NOTE: place the authors' ResNet50_IMAGENET1K_V2.pt under repos/DeepPT/ if"
echo "      available; otherwise torchvision's IMAGENET1K_V2 weights are used"
echo "      (verified equivalent by the harness)."
