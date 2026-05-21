

Generate instance: 

```
python main.py generate [--xml-files FILE[,FILE…]] [--max-phases] [--number-of-processes] [--arrival-rate] [--batch-size] [--min-base-duration] [--max-base-duration] [--min-resource-ratio] [--resource-ratio-center] [--resource-ratio-spread] [--timeout] [--n-resources]
```


Run a single instance: 

```
python main.py run [instance_id] [--objective makespan|flow-time] [--models PDT|PDDT|SDT|SDDT|OODDT|OOPDT|OOPDDT|MSEQCT] [--scarcities 0.0–1.0]
```

Run all instances:

```
python main.py run-all [--objective makespan|flow-time]
```


