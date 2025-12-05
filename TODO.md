# TODO
**Goal**: Low power, always-on doorbell

## MVP
1. run the camera
    - see what the camera sees
2. translate camera events to SNN inputs
    - define (spatiotemporal) receptive fields
3. run SNN (online mode, concurrent threading)
    - define the model/arch
    - collect classified hand gesture data
4. ring the bell (audio)

### Considerations
- RPi capacity to run both SNN and Brian2
- Running Brian2 on event hardware (or find alternative)

### Extensions
- Foveated vison and actuated movement


## Task Delegation
### Mahima
part 2, 3(model), 4
### Niklas
part 1, 3(concurrency)