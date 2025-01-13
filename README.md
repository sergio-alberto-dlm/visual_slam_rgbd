# Visual Odometry Project 

## Enable cuda support 

To use Pytorch with cuda support, install as follows:

```bash
conda uninstall pytorch torchvision
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

## Notes on Submodules

### Cloning with Submodules

Use the `--recurse-submodules` flag to clone the submodules as well:

```bash
git clone --recurse-submodules <repo-url>
```

### Initializing and Updating Submodules Manually

Alternatively, after cloning the repository, submodules can be initialized and updated manually using the following commands:

```bash
git submodule init
git submodule update
```

These steps ensure that all submodules are correctly fetched and set up for use.
