# macOS Installation Guide for napari-neuron-navigator

This guide installs `napari-neuron-navigator` into a standard folder:

```bash
~/repos/napari-neuron-navigator
```

`~` means your macOS home folder, usually `/Users/yourname`.

The project uses Pixi to install Python, napari, and the plugin dependencies. You do not need to create a separate Python virtual environment.

## 1. Open Terminal

Use any one of these methods:

1. Press `Command-Space`, type `Terminal`, and press `Return`.
2. Open Finder, go to `Applications`, then `Utilities`, then double-click `Terminal`.
3. Open Launchpad, search for `Terminal`, and open it.

When Terminal opens, you will see a prompt. The examples below show commands only. Do not type extra prompt characters such as `$` if your Terminal shows them.

## 2. Confirm Git Is Available

This guide assumes Git is already present on the Mac. You can confirm with:

```bash
git --version
```

If macOS asks to install command line developer tools, allow it to install them, then run `git --version` again.

## 3. Create a `repos` Directory in Your Home Folder

In Terminal, run:

```bash
mkdir -p ~/repos
cd ~/repos
pwd
```

The `pwd` command prints the folder you are currently in. It should end with:

```text
/repos
```

For example:

```text
/Users/yourname/repos
```

## 4. Clone the GitHub Repository

The repository is public. You do not need special GitHub permission to download it.

1. Open a web browser.
2. Go to:

   <https://github.com/nimh-dsst/napari-neuron-navigator>

3. Click the green `Code` button.
4. Choose either `HTTPS` or `SSH`.

Use `HTTPS` unless you already know that SSH keys are set up for your GitHub account.

### Recommended: Clone with HTTPS

Make sure Terminal is still in `~/repos`, then run:

```bash
git clone https://github.com/nimh-dsst/napari-neuron-navigator.git
```

Then enter the new folder:

```bash
cd ~/repos/napari-neuron-navigator
```

### Alternative: Clone with SSH

Only use SSH if your GitHub SSH keys are already configured.

```bash
git clone git@github.com:nimh-dsst/napari-neuron-navigator.git
cd ~/repos/napari-neuron-navigator
```

### Confirm the Clone Worked

Run:

```bash
ls
```

You should see files such as:

```text
README.md
MANUAL.MD
pixi.toml
src
docs
```

## 5. Install Pixi

Pixi is the dependency manager used by this project. It installs the right Python and napari environment for you.

Official Pixi installation page:

<https://pixi.prefix.dev/latest/installation/>

In Terminal, run:

```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

When the installer finishes, close Terminal completely and open a new Terminal window. This lets macOS reload your shell settings.

Then check Pixi:

```bash
pixi --version
```

If you see a Pixi version number, Pixi is installed.

### If `pixi` Is Not Found

Close Terminal and open it again, then retry:

```bash
pixi --version
```

If that still does not work, run the command for your shell:

```bash
source ~/.zshrc
```

Most modern Macs use Zsh. If you use Bash instead, run:

```bash
source ~/.bashrc
```

Then retry:

```bash
pixi --version
```

## 6. Launch napari with the Plugin

Go to the repository folder:

```bash
cd ~/repos/napari-neuron-navigator
```

Start napari:

```bash
pixi run napari
```

The first run can take several minutes because Pixi may need to download and install packages.

When napari opens, use the napari `Plugins` menu and open `Neuron Viewer`.

## 7. Put a Launcher Script on the Desktop

The repository includes two macOS-compatible launcher examples:

```text
run_scripts/run_napari.zsh
run_scripts/run_napari.sh
```

The Zsh script is recommended for modern macOS.

### Create a Double-Click Desktop Launcher

Run these commands in Terminal:

```bash
cp ~/repos/napari-neuron-navigator/run_scripts/run_napari.zsh ~/Desktop/Run_napari_Neuron_Navigator.command
chmod +x ~/Desktop/Run_napari_Neuron_Navigator.command
```

You can now double-click `Run_napari_Neuron_Navigator.command` on your Desktop.

If macOS warns that the file cannot be opened because it is from an unidentified developer:

1. Right-click or Control-click the file.
2. Choose `Open`.
3. Confirm that you want to open it.

### Run the Script from Terminal Instead

You can also run:

```bash
~/repos/napari-neuron-navigator/run_scripts/run_napari.zsh
```

or:

```bash
~/repos/napari-neuron-navigator/run_scripts/run_napari.sh
```

## 8. Updating Later

To get the newest public version from GitHub later:

```bash
cd ~/repos/napari-neuron-navigator
git pull
pixi run napari
```

## Troubleshooting

### `git clone` Says the Folder Already Exists

If this command:

```bash
git clone https://github.com/nimh-dsst/napari-neuron-navigator.git
```

says the folder already exists, the repository may already be downloaded. Try:

```bash
cd ~/repos/napari-neuron-navigator
git pull
```

### SSH Clone Fails

If this command fails:

```bash
git clone git@github.com:nimh-dsst/napari-neuron-navigator.git
```

use HTTPS instead:

```bash
git clone https://github.com/nimh-dsst/napari-neuron-navigator.git
```

### Pixi Takes a Long Time the First Time

That is expected. The first `pixi run napari` has to create the environment and download dependencies. Later launches should be faster.

### The Desktop Launcher Cannot Find the Repository

The launcher assumes this exact folder:

```bash
~/repos/napari-neuron-navigator
```

If you cloned the repository somewhere else, either move it to that folder or edit the launcher script to use your actual path.

## Reference Links

- Project repository: <https://github.com/nimh-dsst/napari-neuron-navigator>
- Pixi installation: <https://pixi.prefix.dev/latest/installation/>
- GitHub cloning guide: <https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository>
