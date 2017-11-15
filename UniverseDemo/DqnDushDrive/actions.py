# ('KeyEvent', 'ArrowUp', True)
au = 'ArrowUp'
ad = 'ArrowDown'
al = 'ArrowLeft'
ar = 'ArrowRight'
KEYEVENT = 'KeyEvent'
up_right = [(KEYEVENT, au, True), (KEYEVENT, ad, False), (KEYEVENT, ar, True), (KEYEVENT, al, False)]
up_left = [(KEYEVENT, au, True), (KEYEVENT, ad, False), (KEYEVENT, ar, False), (KEYEVENT, al, True)]
down_left = [(KEYEVENT, au, False), (KEYEVENT, ad, True), (KEYEVENT, ar, False), (KEYEVENT, al, True)]
down_right = [(KEYEVENT, au, False), (KEYEVENT, ad, True), (KEYEVENT, ar, True), (KEYEVENT, al, False)]
up = [(KEYEVENT, au, True), (KEYEVENT, ad, False), (KEYEVENT, ar, False), (KEYEVENT, al, False)]
down = [(KEYEVENT, au, False), (KEYEVENT, ad, True), (KEYEVENT, ar, False), (KEYEVENT, al, False)]
no_op = []
# 6 actions
ACTIONS = [up, up_left, up_right, down, down_right, down_left]
