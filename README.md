## Descrição

Lane Keeping Assist usando SegFormer-B0

## Arquivos de testes

- `data/raw/test_video.mp4`
- `data/raw/video_02.mp4`
- `config/test_video_params.json`
- `config/video_02_params.json`
- `experiments/segformer_b0_final/best.pt`

## Ambiente

Use Python 3.11. Na primeira vez, crie o ambiente virtual e instale as dependencias:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Nas proximas vezes, para ligar/ativar o ambiente virtual, execute:

```powershell
.\.venv\Scripts\Activate.ps1
```

Se o `requirements.txt` for alterado ou se precisar reinstalar as dependências, execute com o ambiente virtual ativo:

```powershell
pip install -r requirements.txt
```

## Executar video_01 / test_video

```powershell
python main.py --video video_01
```

## Executar video_02

```powershell
python main.py --video video_02
```

## Cores

Faixas de parametro:
- azul: faixa esquerda real
- vermelho: faixa direita real
- branco: centro das faixas
  
Faixas de controle:
- ciano: pseudo-lane 20 px a direita da faixa esquerda
- magenta: pseudo-lane 20 px a esquerda da faixa direita

## Exemplo comando

python main.py --video video_02  --pseudo-left-offset 20  --pseudo-right-offset 30 --target left

python main.py --video-path data/raw/video_04.mp4 --config config/test_video_params.json --target center --roi-top-left 0 220 --roi-top-right 848 220 --roi-bottom-left 0 360 --roi-bottom-right 848 360 --display-delay 1

## Câmera em tempo real

python camera_main.py --camera-index 0 --config config/test_video_params.json --target center --camera-width 848 --camera-height 478 --camera-fps 30 --roi-top-left 0 220 --roi-top-right 848 220 --roi-bottom-left 0 360 --roi-bottom-right 848 360 --display-delay 1

