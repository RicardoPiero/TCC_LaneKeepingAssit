# Controle continuo integrado ao runtime LKAS

O `main.py` executa agora:

`SegFormer-B0 -> geometria v3 -> tracking -> target -> erro -> controlador P -> steering`

## Parametros padrao

- `kp = 0.8`
- `deadband_px = 3`
- `smoothing_alpha = 0.25`
- `max_step = 0.08`
- `max_output = 1.0`
- `steering_hold_frames = 10`

Convencao:

- `steering > 0`: corrigir para a esquerda;
- `steering < 0`: corrigir para a direita;
- `steering ~= 0`: manter direcao.

## Teste visual

```powershell
python main.py --video video_02 --target center --display-delay 100
```

O painel mostra:

- `err`: erro lateral em pixels;
- `norm`: erro normalizado pela metade da largura da pista;
- `steer`: comando filtrado entre `-1` e `+1`;
- `ESQUERDA`, `RETO` ou `DIREITA`: interpretacao do steering;
- `HOLD`: ultimo comando mantido durante perda curta do target.

## Targets laterais

```powershell
python main.py --video video_02 --target left --pseudo-left-offset 20 --display-delay 100
```

```powershell
python main.py --video video_02 --target right --pseudo-right-offset 20 --display-delay 100
```

## Exportar dataset final com controle integrado

Nao sobrescreva os CSVs originais de percepcao.

```powershell
python main.py --video video_01 --target center `
    --export-dataset experiments/lkas_runtime_control_video_01.csv
```

```powershell
python main.py --video video_02 --target center `
    --export-dataset experiments/lkas_runtime_control_video_02.csv
```

Os novos CSVs incluem:

- `lane_width_px`;
- `error_normalized`;
- `steering_raw`;
- `steering_filtered`;
- `steering_delta`;
- `steering_direction`;
- `steering_held`;
- `steering_saturated`;
- parametros usados pelo controlador.

Os videos gravados nao respondem ao steering. Esta etapa valida a integracao e a estabilidade do sinal, nao o desempenho em malha fechada do veiculo.
