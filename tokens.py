# huggingface token for automated model downloading. Not required when logged in with huggingface-cli or when using local model checkpoints.
HUGGINGFACE_TOKEN = ""
# token for the discord bot when using discord_bot.py
DISCORD_TOKEN = ""

def get_discord_token():
    # str() encapsulation is done to prevent some python language servers from previewing your token as a literal when hovering get_token()
    return str(DISCORD_TOKEN)

# if a huggingface token is set, it will be returned. If the default value of 'True' is used, diffusers will attempt access via a pre-existing huggingface-cli login
def get_huggingface_token():
    return str(HUGGINGFACE_TOKEN) if len(HUGGINGFACE_TOKEN) > 0 else True

if __name__ == "__main__":
    print("To use this file, edit it with your access token(s) and import the relevant function(s) in your script.")
