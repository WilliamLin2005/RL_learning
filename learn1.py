'''
first program for py_learn
'''

secret_num=42;
guess_time=1;

guess_num=int(input("please enter guessed number: "));

while guess_num!=secret_num and guess_time<5:
    guess_num=int(input("Incorredt,please enter guessed number: "));
    guess_time=guess_time+1;

if guess_num == secret_num:
    print("Correct");
else:
    print("game time up,gameover");
    