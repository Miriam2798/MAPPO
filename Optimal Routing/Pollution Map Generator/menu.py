import pathtest as pt
import pollutionmap as pm
import transformap as tm
import time


def menu():
    print(r"""   _____      _____ ____________________________   
  /     \    /  _  \\______   \______   \_____  \  
 /  \ /  \  /  /_\  \|     ___/|     ___//   |   \ 
/    Y    \/    |    \    |    |    |   /    |    \
\____|__  /\____|__  /____|    |____|   \_______  /
        \/         \/                           \/ v0.2 """)
    time.sleep(1)
    print("1. Simulate Pollution Map \n")
    print("2. Transform Pollution Map \n")
    print("3. Calculate Fastest Route \n")
    print("4. Exit \n")


loop = True
while True:
    menu()
    choice = input("Enter your choice [1-4]: \n")
    if choice == "1":
        pm.main()
    elif choice == "2":
        tm.main()
    elif choice == "3":
        pt.main()
    elif choice == "4":
        loop = False
        break
    else:
        input("Wrong Option. Press any key to try again")