# docs/demo/demo.tape — reproducible source for docs/images/demo.gif
#
# Rebuild:  pip install 'yantrikdb-mcp[onnx]'
#           cd docs/demo && vhs demo.tape   # writes ../images/demo.gif
#
# demo.py drives the real MCP server over stdio; every value on screen is
# the server's own answer.

Output ../images/demo.gif

Require vhs
Require python

Set Shell bash
Set FontFamily "JetBrains Mono"
Set FontSize 21
Set Width 1160
Set Height 900
Set Padding 18
Set Framerate 8
Set TypingSpeed 45ms
Set Theme { "background": "#282c34", "foreground": "#abb2bf", "cursor": "#abb2bf", "selection": "#3e4451", "black": "#282c34", "red": "#e06c75", "green": "#98c379", "yellow": "#e5c07b", "blue": "#61afef", "magenta": "#c678dd", "cyan": "#56b6c2", "white": "#abb2bf", "brightBlack": "#5c6370", "brightRed": "#e06c75", "brightGreen": "#98c379", "brightYellow": "#e5c07b", "brightBlue": "#61afef", "brightMagenta": "#c678dd", "brightCyan": "#56b6c2", "brightWhite": "#ffffff" }

Hide
Type "export PS1='$ ' && clear"
Enter
Show

Sleep 600ms
Type "python demo.py"
Sleep 500ms
Enter
Sleep 24s
