# Install Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Add Homebrew to PATH
eval "$(/opt/homebrew/bin/brew shellenv)"

# Reload your shell configuration
source ~/.bash_profile  # or source ~/.bashrc

# Verify Homebrew is installed
brew --version

# Fix permissions
sudo chown -R $(whoami) /opt/homebrew  # for Apple Silicon

# Update Homebrew
brew update