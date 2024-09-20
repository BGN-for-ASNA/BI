To set up Git, VS Code, Markdown live preview, and Quarto for a streamlined workflow, follow these steps:

### 1. Install Git

Git is essential for version control. Install it on your machine:

- **Windows**: Download from [git-scm.com](https://git-scm.com/) and follow the installer steps.
- **Mac**: Run the following command in your terminal:
  ```bash
  brew install git
  ```
- **Linux**: Use the appropriate package manager for your distribution:
  ```bash
  sudo apt-get install git  # For Ubuntu/Debian
  ```

Once installed, verify with:
```bash
git --version
```

#### Setup Git:

Set up your username and email:
```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

### 2. Install VS Code

1. Download VS Code from the [official website](https://code.visualstudio.com/).
2. Follow the instructions to install it for your OS.

### 3. Install Quarto

Quarto is used for creating high-quality, reproducible documents. Install Quarto by following these steps:

1. Download Quarto from the [official site](https://quarto.org/docs/get-started/).
2. Install Quarto following your OS-specific instructions:
   - **Windows/Mac**: Run the installer package.
   - **Linux**: Use the following terminal command:
     ```bash
     sudo apt-get install quarto
     ```

Verify the installation:
```bash
quarto --version
```

### 4. Setup VS Code extensions
 Open VS Code and install some extensions for a better Markdown and Quarto experience:
   - **GitLens** (for Git integration).
   - **Git Graph** (for easy commits).
   - **Markdown All in One** (for enhanced Markdown editing and live preview).
   - **Quarto** (for Quarto-specific support).

### 5. Initialize a Git Repository
1.  Create a folder
2.  Invscode in ```file->open folder``` select your folder.
3.  Opent terminal with ```ctrl+j``` or ```cmd+j```
4.  Clone branch with command : ``` git clone -b BI_numpyro https://github.com/BGN-for-ASNA/BI.git```
5.  Documentation is in  ```Documentation``` folder
6.  

### 6. Commit and push your changes
In vscode you should have the following icon 
![](Screenshot1.png)
Give a name to the commit:
![](S2.png)


