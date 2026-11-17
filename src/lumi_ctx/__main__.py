import sys

from .cli import main

# A guarda é necessária: os processos "spawn" do multiprocessing reimportam este
# módulo como __mp_main__ e não podem rodar a CLI de novo.
if __name__ == "__main__":
    sys.exit(main())
