# SegFormer-B0 — Runtime LKA

Branch enxuta para executar o Lane Keeping Assist usando somente o SegFormer-B0, a geometria v3, tracking temporal e as tres pseudo-lanes de guiagem.

## Arquivos locais que precisam ser copiados

Esta branch nao versiona videos, configuracoes ajustadas localmente nem o checkpoint treinado. Depois de baixar a branch, copie do projeto anterior:

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

Se o `requirements.txt` for alterado ou se precisar reinstalar as dependencias, execute com o ambiente virtual ativo:

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

## Pipeline

`video -> ROI -> BEV -> CLAHE -> SegFormer-B0 -> mascara -> geometria v3 -> tracking -> lanes -> offset/comando`

Parametros padrao validados:

- entrada neural: `384x256`
- historico temporal: `8` frames
- hold: `10` frames
- pseudo-lanes internas: `20 px`

## Cores

- azul: faixa esquerda real
- vermelho: faixa direita real
- ciano: pseudo-lane 20 px a direita da faixa esquerda
- branco: centro geometrico da pista
- magenta: pseudo-lane 20 px a esquerda da faixa direita
- verde: area entre as duas faixas

## Opcoes uteis

```powershell
python main.py --video video_02 --pseudo-offset 20 --display-delay 100
```

Para um video customizado:

```powershell
python main.py --video-path data/raw/meu_video.mp4 --config config/meu_video.json
```

Para salvar a visualizacao:

```powershell
python main.py --video video_02 --output-video experiments/segformer_video_02.mp4
```

`Q` ou `ESC` encerra; `ESPACO` pausa/continua.
