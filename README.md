# Enhancing Feasibility of Cost-Optimal Energy Systems in Surrogate-Assisted Design Optimization

## Referencing
If you use our repository or any part of it, please cite [10.2139/ssrn.6477860](https://doi.org/10.2139/ssrn.6477860).
The full information on the publication is also shown below.

**Enhancing Feasibility of Cost-Optimal Energy Systems in Surrogate-Assisted Design
Optimization**  
B. Jürgens, K. Reents, N. Baumeister, H. Seele, H. Schricker, N. von der Assen (2026).  
*Preprint on SSRN.*  
DOI: [10.2139/ssrn.6477860](https://doi.org/10.2139/ssrn.6477860)

For the data of the Industrial energy system we used in this publication, please also refer to
[Reinert et al. (2023)](https://doi.org/10.1016/j.compchemeng.2023.108176).

## Licensing
- The source code in this repository is licensed under the MIT license, for more information please refer to the license file.
- The data in `data/` is licensed under Creative Commons Attribution 4.0 (CC BY 4.0).

## Installation
- Python version on LTT working station: 3.12.5
- Python version on RWTH HPC: 3.12.3

Open the cmd-console. <br>

**Step 0:** Make sure that you have the same Python version installed as stated above!
Otherwise, you may have significant trouble with package dependencies.

**Step 1:** Navigate to the folder, where the repository should be cloned.
You can change directory with `cd FOLDERNAME` and make new directories (folders) with `mkdir FOLDERNAME`.
```
cd PATH/TO/THE/FOLDER/
```
**Step 2:** Clone project (via SSH or https, the tutorial for https is provided below).
**Warning:** Keep in mind while entering your password, that the keyboard-format might be English **or** German.
**Note:** If you used Token Access, the username is the Token_name (e.g. my_Token) and the password is the Token password generated online.
```
git init
git clone https://git-ce.rwth-aachen.de/ltt/opt/deinProjekt.git
```
**Step 3:** Change to your branch:
```
git checkout origin/your_branch_name
```
**Step 4:** Change to the path, where the environment should be saved (e.g. the ROOT directory, where .git lies). <br>
Make a new folder called venv (`mkdir venv`) and navigate into the new folder (`cd venv`).


**Step 5:** Create Pip environment with a name (e.g., 'venv_sado'):
```
python -m venv venv_sado
```
**Step 6:** Activate the environment.
```
.\venv_sado\Scripts\activate
```
**Step 7:** Make sure you are in the ROOT directory (where .git lies). Install the package with
```
pip install ".[dev]"
```
This might take a while (about 10 minutes). If other packages are missing, install them with:
```
pip install PackageName
```
A specific version can be installed with:
```
pip install PackageName==version
```
**Step 8:** To use jupyter notebook using your own virtual environment, you must activate the kernel while your virtual environment is activated. If you don't do the following, your venv won't show up in the "Select Kernel" dialog:
```
python -m ipykernel install --user --name=venv_sado
```

## Additional info
If you have any questions, please contact: boyung.juergens@ltt.rwth-aachen.de

Contributers to this repository: Jürgens, Boyung; Reents, Konrad; Baumeister, Nils; Seele, Hagen; Schricker, Hendrik; von der Assen, Niklas
