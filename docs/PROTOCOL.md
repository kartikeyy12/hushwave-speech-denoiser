# Serial protocol

The page (or any host) and the ESP32 exchange frames over the board's USB serial port.
`src/hushwave/mcu.py`, `demo/web/protocol.js` and `firmware/hushwave_esp32/hushwave_protocol.cpp` implement it.

```
A5 5A | type | seq | length (u16, little endian) | payload | crc16 (little endian)
```

`crc16` is CRC-16/CCITT-FALSE (polynomial 0x1021, initial value 0xFFFF, no reflection) over type, seq, length and payload.
The check value of the ASCII string `123456789` is `0x29B1`. All multi-byte fields are little endian. The payload is at most 1024 bytes.

| Type | Name | Direction | Payload |
|---|---|---|---|
| 0x01 | AUDIO_IN | host to board | 256 samples, 16-bit signed PCM (512 bytes) |
| 0x81 | AUDIO_OUT | board to host | processing time in microseconds (u16), status (u16, zero), 256 samples of 16-bit PCM |
| 0x02 | HELLO | host to board | none |
| 0x82 | HELLO_REPLY | board to host | version (u8), sample rate (u16), block (u16), delay in samples (u16), firmware name (ASCII) |
| 0x03 | RESET | host to board | none; forgets the noise estimate and all buffers |
| 0x83 | RESET_ACK | board to host | none |

Rules
- Samples are scaled by 32768. The host clips to the 16-bit range; so does the board.
- The board answers every valid AUDIO_IN with an AUDIO_OUT carrying the same `seq`, in order.
- The first output sample corresponds to the input sample 511 positions earlier (the algorithmic delay).
- A frame with a wrong CRC, an impossible length or an unknown type is dropped without an answer. The parser looks for the next sync marker, so a damaged frame costs one block of 16 ms and nothing else.
- The host keeps up to four blocks in flight and treats a block as lost after 500 ms.
- Bytes that are not part of a frame (for example the boot messages of the board) are ignored.

Link load at 921 600 baud (10 bits per byte): 520 bytes per block upstream and 524 bytes downstream, every 16 ms.
