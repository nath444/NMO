import os

def show_ascii_banner():
    ascii_art = r'''
                                                                                                   
                  +---*                                                                            
              @-#%=-+%@-#@:                                                                        
             @%*--:%%@@@@@+=-                                                                      
            @%*=-::::::--=#@@:                                                                     
           =@+%--==-::-++=+@@#                                                                     
           @@+%+=*@*:-@%++##@@*                                                                    
          ::%%=-:::+:-*:--=%%%-         @@@        @@@  @@@@         @@@    @@@@@@@@@@@            
          --@%#=---%-=%===+@%@-         @@@@@      @@@  @@@@@@    @@@@@@  @@@@@@@@@@@@@@@          
           %#@%+=--=@@*=+*#@@##         @@@ @@@@   @@@  @@@ @@@@@@@@ @@@  @@@@        @@@          
            *@@#==-:::*+**#@@*+         @@@   @@@@ @@@  @@@   @@@@   @@@  @@@@        @@@          
               @+=-=--=+*%              @@@     @@@@@@  @@@          @@@  @@@@@@@@@@@@@@@          
               @@%=--=*#@@              @@@        @@@  @@@          @@@    @@@@@@@@@@@            
               @=%@@@@@%#@                                                                         
              %%==#@@@%#*@@@                                                                       
             @%%*+*#@@%%%@@@@                                                                      
             =--*%@@@@@@%++#                                                                       
               ::::*@%----                                                                         
                  +:::+                                                                            
                                                                                                   
    '''
    print(ascii_art)

def main():
    os.system('clear' if os.name != 'nt' else 'cls')
    show_ascii_banner()

    print("\nBienvenue dans NMO (NTH Music Organizer).\n")
    print("Que souhaitez-vous faire ?")
    print("1. Analyser")
    print("2. Classifier")

    choix = input("\nEntrez 1 ou 2 : ").strip()

    if choix == '1':
        os.system("python3 analyse.py")
    elif choix == '2':
        os.system("python3 classifier.py")
    else:
        print("\nChoix invalide.")

if __name__ == "__main__":
    main()
