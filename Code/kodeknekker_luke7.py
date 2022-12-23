morse = {
    "/": " ",
    ".-" : "a",
    "-...": "b",
    "-.-.": "c",
    "-..": "d",
    ".": "e",
    "..-.": "f",
    "--.": "g",
    "....": "h",
    ".." : "i",
    ".---" : "j",
    "-.-" : "k",
    ".-.." : "l",
    "--" : "m",
    "-." : "n",
    "---" : "o",
    ".--." : "p",
    "--.-" : "q",
    ".-." : "r",
    "..." : "s",
    "-" : "t",
    "..-" : "u",
    "...-" : "v",
    ".--" : "w",
    "-..-" : "x",
    "-.--" : "y",
    "--.." : "z",
    "-----" : "0",
    ".----" : "1",
    "..---" : "2",
    "...--" : "3",
    "....-" : "4",
    "....." : "5",
    "-...." : "6",
    "--..." : "7",
    "---.." : "8",
    "----." : "9",
    ".-.-.-" : ".",
    "--..--" : ",",
    "..--.." : "?",
    ".----." : "'",
    "-.-.--" : "!",
    ".--.-" : "å",
    ".-.-" : "æ",
    "---." : "ø",
    "---..." : ":",
    "-..-." : "/",
    ".-..-." : "\"",
    "-....-" : "-",
    ".....--" : "#",
    "-.--.-" : ")"
}


def decrypt():
    code = input()

    # Bytt til ordentlig morse
    code = code.replace("Ås", "-").replace("I", ".")

    # Del opp i tegn
    code = code.split(" ")

    solved = ""
    for char in code:
        if not char in morse:
            print(f"{char} does not exist in alphabet")
            quit()
        solved += morse[char]

    print(solved)


def encrypt():
    solved = input()

    reverse_dict = {}
    for it, val in morse.items():
        reverse_dict[val] = it

    code = " ".join([reverse_dict[ch] for ch in solved])

    code = code.replace(".", "I").replace("-", "Ås")

    print(code)


decrypt()