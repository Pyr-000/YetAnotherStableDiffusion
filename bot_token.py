def get_token():
    # str() encapsulation is done to prevent some python language servers from previewing your token as a literal when hovering get_token()
    return str("YOUR TOKEN HERE")

if __name__ == "__main__":
    print("To use bot_token.py, import bot_token, and call get_token().")