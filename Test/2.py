def truncate_list(lst, length):
    new_lst = []
    for item in lst:
        if len(item) <= length:
            new_lst.append(item)
        else:
            for i in range(0, len(item), length):
                new_lst.append(item[i:i+length])
    return new_lst

label=['PUSH1 0x80 PUSH1 0x40 MSTORE CALLVALUE DUP1 ISZERO PUSH2 0x10 JUMPI PUSH1 0x0 DUP1 REVERT JUMPDEST POP PUSH1 0x4 CALLDATASIZE LT PUSH2 0xa9 JUMPI PUSH1 0x0 CALLDATALOAD PUSH1 0xe0 SHR DUP1 PUSH4 0x39509351 GT PUSH2 0x71 JUMPI DUP1 PUSH4 0x39509351 EQ PUSH2 0x129 JUMPI DUP1 PUSH4 0x70a08231 EQ PUSH2 0x13c JUMPI DUP1 PUSH4 0x95d89b41 EQ PUSH2 0x14f JUMPI DUP1 PUSH4 0xa457c2d7 EQ PUSH2 0x157 JUMPI DUP1 PUSH4 0xa9059cbb EQ PUSH2 0x16a JUMPI DUP1 PUSH4 0xdd62ed3e EQ PUSH2 0x17d JUMPI PUSH2 0xa9 JUMP JUMPDEST DUP1 PUSH4 0x6fdde03 EQ PUSH2 0xae JUMPI DUP1 PUSH4 0x95ea7b3 EQ PUSH2 0xcc JUMPI DUP1 PUSH4 0x18160ddd EQ PUSH2 0xec JUMPI DUP1 PUSH4 0x23b872dd EQ PUSH2 0x101 JUMPI DUP1 PUSH4 0x313ce567 EQ PUSH2 0x114 JUMPI JUMPDEST PUSH1 0x0 DUP1 REVERT JUMPDEST PUSH2 0xb6 PUSH2 0x190 JUMP JUMPDEST PUSH1 0x40 MLOAD PUSH2 0xc3 SWAP2 SWAP1 PUSH2 0x6df JUMP JUMPDEST PUSH1 0x40 MLOAD DUP1 SWAP2 SUB SWAP1 RETURN JUMPDEST PUSH2 0xdf PUSH2 0xda CALLDATASIZE PUSH1 0x4 PUSH2 0x6ab JUMP JUMPDEST PUSH2 0x222 JUMP JUMPDEST PUSH1 0x40 MLOAD PUSH2 0xc3 SWAP2 SWAP1 PUSH2 0x6d4 JUMP JUMPDEST PUSH2 0xf4 PUSH2 0x23f JUMP JUMPDEST PUSH1 0x40 MLOAD PUSH2 0xc3 SWAP2 SWAP1 PUSH2 0x913 JUMP JUMPDEST PUSH2 0xdf PUSH2 0x10f CALLDATASIZE PUSH1 0x4 PUSH2 0x670 JUMP JUMPDEST PUSH2 0x245 JUMP JUMPDEST PUSH2 0x11c PUSH2 0x2e5 JUMP JUMPDEST PUSH1 0x40 MLOAD PUSH2 0xc3 SWAP2 SWAP1 PUSH2 0x91c JUMP JUMPDEST PUSH2 0xdf PUSH2 0x137 CALLDATASIZE PUSH1 0x4 PUSH2 0x6ab JUMP JUMPDEST PUSH2 0x2ea JUMP JUMPDEST PUSH2 0xf4 PUSH2 0x14a CALLDATASIZE PUSH1 0x4 PUSH2 0x61d JUMP JUMPDEST PUSH2 0x339 JUMP JUMPDEST PUSH2 0xb6 PUSH2 0x358 JUMP JUMPDEST PUSH2 0xdf PUSH2 0x165 CALLDATASIZE PUSH1 0x4 PUSH2 0x6ab JUMP JUMPDEST PUSH2 0x367 JUMP JUMPDEST PUSH2 0xdf PUSH2 0x178 CALLDATASIZE PUSH1 0x4 PUSH2 0x6ab JUMP JUMPDEST PUSH2 0x3e2 JUMP JUMPDEST PUSH2 0xf4 PUSH2 0x18b CALLDATASIZE PUSH1 0x4 PUSH2 0x63e JUMP JUMPDEST PUSH2 0x3f6 JUMP JUMPDEST PUSH1 0x60 PUSH1 0x3 DUP1 SLOAD PUSH2 0x19f SWAP1 PUSH2 0x959 JUMP JUMPDEST DUP1 PUSH1 0x1f ADD PUSH1 0x20 DUP1 SWAP2 DIV MUL PUSH1 0x20 ADD PUSH1 0x40 MLOAD SWAP1 DUP2 ADD PUSH1 0x40 MSTORE DUP1 SWAP3 SWAP2 SWAP1 DUP2 DUP2 MSTORE PUSH1 0x20 ADD DUP3 DUP1 SLOAD PUSH2 0x1cb SWAP1 PUSH2 0x959 JUMP JUMPDEST DUP1 ISZERO PUSH2 0x218 JUMPI DUP1 PUSH1 0x1f LT PUSH2 0x1ed JUMPI PUSH2 0x100 DUP1 DUP4 SLOAD DIV MUL DUP4 MSTORE SWAP2 PUSH1 0x20 ADD SWAP2 PUSH2 0x218 JUMP JUMPDEST DUP3 ADD SWAP2 SWAP1 PUSH1 0x0 MSTORE PUSH1 0x20 PUSH1 0x0 SHA3 SWAP1 JUMPDEST DUP2 SLOAD DUP2 MSTORE SWAP1 PUSH1 0x1 ADD SWAP1 PUSH1 0x20 ADD DUP1 DUP4 GT PUSH2 0x1fb JUMPI DUP3 SWAP1 SUB PUSH1 0x1f AND DUP3 ADD SWAP2 JUMPDEST POP POP POP POP POP SWAP1 POP SWAP1 JUMP JUMPDEST PUSH1 0x0 PUSH2 0x236 PUSH2 0x22f PUSH2 0x421 JUMP JUMPDEST DUP5 DUP5 PUSH2 0x425 JUMP JUMPDEST POP PUSH1 0x1 SWAP3 SWAP2 POP POP JUMP JUMPDEST PUSH1 0x2 SLOAD SWAP1 JUMP JUMPDEST PUSH1 0x0 PUSH2 0x252 DUP5 DUP5 DUP5 PUSH2 0x4d9 JUMP JUMPDEST PUSH1 0x1 PUSH1 0x1 PUSH1 0xa0 SHL SUB DUP5 AND PUSH1 0x0 SWAP1 DUP2 MSTORE PUSH1 0x1 PUSH1 0x20 MSTORE PUSH1 0x40 DUP2 SHA3 DUP2 PUSH2 0x273 PUSH2 0x421 JUMP JUMPDEST PUSH1 0x1 PUSH1 0x1 PUSH1 0xa0 SHL SUB AND PUSH1 0x1 PUSH1 0x1 PUSH1 0xa0 SHL SUB AND DUP2 MSTORE PUSH1 0x20 ADD SWAP1 DUP2 MSTORE PUSH1 0x20 ADD PUSH1 0x0 SHA3 SLOAD SWAP1 POP DUP3 DUP2 LT ISZERO PUSH2 0x2bf JUMPI PUSH1 0x40 MLOAD PUSH3 0x461bcd PUSH1 0xe5 SHL DUP2 MSTORE PUSH1 0x4 ADD PUSH2 0x2b6 SWAP1 PUSH2 0x7fd JUMP JUMPDEST PUSH1 0x40 MLOAD DUP1 SWAP2 SUB SWAP1 REVERT JUMPDEST PUSH2 0x2da DUP6 PUSH2 0x2cb PUSH2 0x421 JUMP JUMPDEST PUSH2 0x2d5 DUP7 DUP6 PUSH2 0x942 JUMP JUMPDEST PUSH2 0x425 JUMP JUMPDEST POP PUSH1 0x1 SWAP5 SWAP4 POP POP POP POP JUMP JUMPDEST PUSH1 0x12 SWAP1 JUMP JUMPDEST PUSH1 0x0 PUSH2 0x236 PUSH2 0x2f7 PUSH2 0x421 JUMP JUMPDEST DUP5 DUP5 PUSH1 0x1 PUSH1 0x0 PUSH2 0x305 PUSH2 0x421 JUMP JUMPDEST PUSH1 0x1 PUSH1 0x1 PUSH1 0xa0 SHL SUB SWAP1 DUP2 AND DUP3 MSTORE PUSH1 0x20 DUP1 DUP4 ADD SWAP4 SWAP1 SWAP4 MSTORE PUSH1 0x40 SWAP2 DUP3 ADD PUSH1 0x0 SWAP1 DUP2 SHA3 SWAP2 DUP12 AND DUP2 MSTORE SWAP3 MSTORE SWAP1 SHA3 SLOAD PUSH2 0x2d5 SWAP2 SWAP1 PUSH2 0x92a JUMP JUMPDEST PUSH1 0x1 PUSH1 0x1 PUSH1 0xa0 SHL SUB DUP2 AND PUSH1 0x0 SWAP1 DUP2 MSTORE PUSH1 0x20 DUP2 SWAP1 MSTORE PUSH1 0x40 SWAP1 SHA3 SLOAD JUMPDEST SWAP2 SWAP1 POP JUMP JUMPDEST PUSH1 0x60 PUSH1 0x4 DUP1 SLOAD PUSH2 0x19f SWAP1 PUSH2 0x959 JUMP JUMPDEST PUSH1 0x0 DUP1 PUSH1 0x1 PUSH1 0x0 PUSH2 0x376 PUSH2 0x421 JUMP JUMPDEST PUSH1 0x1 PUSH1 0x1 PUSH1 0xa0 SHL SUB SWAP1 DUP2 AND DUP3 MSTORE PUSH1 0x20 DUP1 DUP4 ADD SWAP4 SWAP1 SWAP4 MSTORE PUSH1 0x40 SWAP2 DUP3 ADD PUSH1 0x0 SWAP1 DUP2 SHA3 SWAP2 DUP9 AND DUP2 MSTORE SWAP3 MSTORE SWAP1 SHA3 SLOAD SWAP1 POP DUP3 DUP2 LT ISZERO PUSH2 0x3c2 JUMPI PUSH1 0x40 MLOAD PUSH3 0x461bcd PUSH1 0xe5 SHL DUP2 MSTORE PUSH1 0x4 ADD PUSH2 0x2b6 SWAP1 PUSH2 0x8ce JUMP JUMPDEST PUSH2 0x3d8 PUSH2 0x3cd PUSH2 0x421 JUMP JUMPDEST DUP6 PUSH2 0x2d5 DUP7 DUP6 PUSH2 0x942 JUMP JUMPDEST POP PUSH1 0x1 SWAP4 SWAP3 POP POP POP JUMP JUMPDEST PUSH1 0x0 PUSH2 0x236 PUSH2 0x3ef PUSH2 0x421 JUMP JUMPDEST DUP5 DUP5 PUSH2 0x4d9 JUMP JUMPDEST PUSH1 0x1 PUSH1 0x1 PUSH1 0xa0 SHL SUB SWAP2 DUP3 AND PUSH1 0x0 SWAP1 DUP2 MSTORE PUSH1 0x1 PUSH1 0x20 SWAP1 DUP2 MSTORE PUSH1 0x40 DUP1 DUP4 SHA3 SWAP4 SWAP1 SWAP5 AND DUP3 MSTORE SWAP2 SWAP1 SWAP2 MSTORE SHA3 SLOAD SWAP1 JUMP JUMPDEST CALLER SWAP1 JUMP JUMPDEST PUSH1 0x1 PUSH1 0x1 PUSH1 0xa0 SHL SUB DUP4 AND PUSH2 0x44b JUMPI PUSH1 0x40 MLOAD PUSH3 0x461bcd PUSH1 0xe5 SHL DUP2 MSTORE PUSH1 0x4 ADD PUSH2 0x2b6 SWAP1 PUSH2 0x88a JUMP JUMPDEST PUSH1 0x1 PUSH1 0x1 PUSH1 0xa0 SHL SUB DUP3 AND PUSH2 0x471 JUMPI PUSH1 0x40 MLOAD PUSH3 0x461bcd PUSH1 0xe5 SHL DUP2 MSTORE PUSH1 0x4 ADD PUSH2 0x2b6 SWAP1 PUSH2 0x775 JUMP JUMPDEST PUSH1 0x1 PUSH1 0x1 PUSH1 0xa0 SHL SUB DUP1 DUP5 AND PUSH1 0x0 DUP2 DUP2 MSTORE PUSH1 0x1 PUSH1 0x20 SWAP1 DUP2 MSTORE PUSH1 0x40 DUP1 DUP4 SHA3 SWAP5 DUP8 AND DUP1 DUP5 MSTORE SWAP5 SWAP1 SWAP2 MSTORE SWAP1 DUP2 SWAP1 SHA3 DUP5 SWAP1 SSTORE MLOAD PUSH32 0x8c5be1e5ebec7d5bd14f71427d1e84f3dd0314c0f7b2291e5b200ac8c7c3b925 SWAP1 PUSH2 0x4cc SWAP1 DUP6 SWAP1 PUSH2 0x913 JUMP JUMPDEST PUSH1 0x40 MLOAD DUP1 SWAP2 SUB SWAP1 LOG3 POP POP POP JUMP JUMPDEST PUSH1 0x1 PUSH1 0x1 PUSH1 0xa0 SHL SUB DUP4 AND PUSH2 0x4ff JUMPI PUSH1 0x40 MLOAD PUSH3 0x461bcd PUSH1 0xe5 SHL DUP2 MSTORE PUSH1 0x4 ADD PUSH2 0x2b6 SWAP1 PUSH2 0x845 JUMP JUMPDEST PUSH1 0x1 PUSH1 0x1 PUSH1 0xa0 SHL SUB DUP3 AND PUSH2 0x525 JUMPI PUSH1 0x40 MLOAD PUSH3 0x461bcd PUSH1 0xe5 SHL DUP2 MSTORE PUSH1 0x4 ADD PUSH2 0x2b6 SWAP1 PUSH2 0x732 JUMP JUMPDEST PUSH2 0x530 DUP4 DUP4 DUP4 PUSH2 0x601 JUMP JUMPDEST PUSH1 0x1 PUSH1 0x1 PUSH1 0xa0 SHL SUB DUP4 AND PUSH1 0x0 SWAP1 DUP2 MSTORE PUSH1 0x20 DUP2 SWAP1 MSTORE PUSH1 0x40 SWAP1 SHA3 SLOAD DUP2 DUP2 LT ISZERO PUSH2 0x569 JUMPI PUSH1 0x40 MLOAD PUSH3 0x461bcd PUSH1 0xe5 SHL DUP2 MSTORE PUSH1 0x4 ADD PUSH2 0x2b6 SWAP1 PUSH2 0x7b7 JUMP JUMPDEST PUSH2 0x573 DUP3 DUP3 PUSH2 0x942 JUMP JUMPDEST PUSH1 0x1 PUSH1 0x1 PUSH1 0xa0 SHL SUB DUP1 DUP7 AND PUSH1 0x0 SWAP1 DUP2 MSTORE PUSH1 0x20 DUP2 SWAP1 MSTORE PUSH1 0x40 DUP1 DUP3 SHA3 SWAP4 SWAP1 SWAP4 SSTORE SWAP1 DUP6 AND DUP2 MSTORE SWAP1 DUP2 SHA3 DUP1 SLOAD DUP5 SWAP3 SWAP1 PUSH2 0x5a9 SWAP1 DUP5 SWAP1 PUSH2 0x92a JUMP JUMPDEST SWAP3 POP POP DUP2 SWAP1 SSTORE POP DUP3 PUSH1 0x1 PUSH1 0x1 PUSH1 0xa0 SHL SUB AND DUP5 PUSH1 0x1 PUSH1 0x1 PUSH1 0xa0 SHL SUB AND PUSH32 0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef DUP5 PUSH1 0x40 MLOAD PUSH2 0x5f3 SWAP2 SWAP1 PUSH2 0x913 JUMP JUMPDEST PUSH1 0x40 MLOAD DUP1 SWAP2 SUB SWAP1 LOG3 POP POP POP POP JUMP JUMPDEST POP POP POP JUMP JUMPDEST DUP1 CALLDATALOAD PUSH1 0x1 PUSH1 0x1 PUSH1 0xa0 SHL SUB DUP2 AND DUP2 EQ PUSH2 0x353 JUMPI PUSH1 0x0 DUP1 REVERT JUMPDEST PUSH1 0x0 PUSH1 0x20 DUP3 DUP5 SUB SLT ISZERO PUSH2 0x62e JUMPI DUP1 DUP2 REVERT JUMPDEST PUSH2 0x637 DUP3 PUSH2 0x606 JUMP JUMPDEST SWAP4 SWAP3 POP POP POP JUMP JUMPDEST PUSH1 0x0 DUP1 PUSH1 0x40 DUP4 DUP6 SUB SLT ISZERO PUSH2 0x650 JUMPI DUP1 DUP2 REVERT JUMPDEST PUSH2 0x659 DUP4 PUSH2 0x606 JUMP JUMPDEST SWAP2 POP PUSH2 0x667 PUSH1 0x20 DUP5 ADD PUSH2 0x606 JUMP JUMPDEST SWAP1 POP SWAP3 POP SWAP3 SWAP1 POP JUMP JUMPDEST PUSH1 0x0 DUP1 PUSH1 0x0 PUSH1 0x60 DUP5 DUP7 SUB SLT ISZERO PUSH2 0x684 JUMPI DUP1 DUP2 REVERT JUMPDEST PUSH2 0x68d DUP5 PUSH2 0x606 JUMP JUMPDEST SWAP3 POP PUSH2 0x69b PUSH1 0x20 DUP6 ADD PUSH2 0x606 JUMP JUMPDEST SWAP2 POP PUSH1 0x40 DUP5 ADD CALLDATALOAD SWAP1 POP SWAP3 POP SWAP3 POP SWAP3 JUMP JUMPDEST PUSH1 0x0 DUP1 PUSH1 0x40 DUP4 DUP6 SUB SLT ISZERO PUSH2 0x6bd JUMPI DUP2 DUP3 REVERT JUMPDEST PUSH2 0x6c6 DUP4 PUSH2 0x606 JUMP JUMPDEST SWAP5 PUSH1 0x20 SWAP4 SWAP1 SWAP4 ADD CALLDATALOAD SWAP4 POP POP POP JUMP JUMPDEST SWAP1 ISZERO ISZERO DUP2 MSTORE PUSH1 0x20 ADD SWAP1 JUMP JUMPDEST PUSH1 0x0 PUSH1 0x20 DUP1 DUP4 MSTORE DUP4 MLOAD DUP1 DUP3 DUP6 ADD MSTORE DUP3 JUMPDEST DUP2 DUP2 LT ISZERO PUSH2 0x70b JUMPI DUP6 DUP2 ADD DUP4 ADD MLOAD DUP6 DUP3 ADD PUSH1 0x40 ADD MSTORE DUP3 ADD PUSH2 0x6ef JUMP JUMPDEST DUP2 DUP2 GT ISZERO PUSH2 0x71c JUMPI DUP4 PUSH1 0x40 DUP4 DUP8 ADD ADD MSTORE JUMPDEST POP PUSH1 0x1f ADD PUSH1 0x1f NOT AND SWAP3 SWAP1 SWAP3 ADD PUSH1 0x40 ADD SWAP4 SWAP3 POP POP POP JUMP JUMPDEST PUSH1 0x20 DUP1 DUP3 MSTORE PUSH1 0x23 SWAP1 DUP3 ADD MSTORE PUSH32 0x45524332303a207472616e7366657220746f20746865207a65726f2061646472 PUSH1 0x40 DUP3 ADD MSTORE PUSH3 0x657373 PUSH1 0xe8 SHL PUSH1 0x60 DUP3 ADD MSTORE PUSH1 0x80 ADD SWAP1 JUMP JUMPDEST PUSH1 0x20 DUP1 DUP3 MSTORE PUSH1 0x22 SWAP1 DUP3 ADD MSTORE PUSH32 0x45524332303a20617070726f766520746f20746865207a65726f206164647265 PUSH1 0x40 DUP3 ADD MSTORE PUSH2 0x7373 PUSH1 0xf0 SHL PUSH1 0x60 DUP3 ADD MSTORE PUSH1 0x80 ADD SWAP1 JUMP JUMPDEST PUSH1 0x20 DUP1 DUP3 MSTORE PUSH1 0x26 SWAP1 DUP3 ADD MSTORE PUSH32 0x45524332303a207472616e7366657220616d6f756e7420657863656564732062 PUSH1 0x40 DUP3 ADD MSTORE PUSH6 0x616c616e6365 PUSH1 0xd0 SHL PUSH1 0x60 DUP3 ADD MSTORE PUSH1 0x80 ADD SWAP1 JUMP JUMPDEST PUSH1 0x20 DUP1 DUP3 MSTORE PUSH1 0x28 SWAP1 DUP3 ADD MSTORE PUSH32 0x45524332303a207472616e7366657220616d6f756e7420657863656564732061 PUSH1 0x40 DUP3 ADD MSTORE PUSH8 0x6c6c6f77616e6365 PUSH1 0xc0 SHL PUSH1 0x60 DUP3 ADD MSTORE PUSH1 0x80 ADD SWAP1 JUMP JUMPDEST PUSH1 0x20 DUP1 DUP3 MSTORE PUSH1 0x25 SWAP1 DUP3 ADD MSTORE PUSH32 0x45524332303a207472616e736665722066726f6d20746865207a65726f206164 PUSH1 0x40 DUP3 ADD MSTORE PUSH5 0x6472657373 PUSH1 0xd8 SHL PUSH1 0x60 DUP3 ADD MSTORE PUSH1 0x80 ADD SWAP1 JUMP JUMPDEST PUSH1 0x20 DUP1 DUP3 MSTORE PUSH1 0x24 SWAP1 DUP3 ADD MSTORE PUSH32 0x45524332303a20617070726f76652066726f6d20746865207a65726f20616464 PUSH1 0x40 DUP3 ADD MSTORE PUSH4 0x72657373 PUSH1 0xe0 SHL PUSH1 0x60 DUP3 ADD MSTORE PUSH1 0x80 ADD SWAP1 JUMP JUMPDEST PUSH1 0x20 DUP1 DUP3 MSTORE PUSH1 0x25 SWAP1 DUP3 ADD MSTORE PUSH32 0x45524332303a2064656372656173656420616c6c6f77616e63652062656c6f77 PUSH1 0x40 DUP3 ADD MSTORE PUSH5 0x207a65726f PUSH1 0xd8 SHL PUSH1 0x60 DUP3 ADD MSTORE PUSH1 0x80 ADD SWAP1 JUMP JUMPDEST SWAP1 DUP2 MSTORE PUSH1 0x20 ADD SWAP1 JUMP JUMPDEST PUSH1 0xff SWAP2 SWAP1 SWAP2 AND DUP2 MSTORE PUSH1 0x20 ADD SWAP1 JUMP JUMPDEST PUSH1 0x0 DUP3 NOT DUP3 GT ISZERO PUSH2 0x93d JUMPI PUSH2 0x93d PUSH2 0x994 JUMP JUMPDEST POP ADD SWAP1 JUMP JUMPDEST PUSH1 0x0 DUP3 DUP3 LT ISZERO PUSH2 0x954 JUMPI PUSH2 0x954 PUSH2 0x994 JUMP JUMPDEST POP SUB SWAP1 JUMP JUMPDEST PUSH1 0x2 DUP2 DIV PUSH1 0x1 DUP3 AND DUP1 PUSH2 0x96d JUMPI PUSH1 0x7f DUP3 AND SWAP2 POP JUMPDEST PUSH1 0x20 DUP3 LT DUP2 EQ ISZERO PUSH2 0x98e JUMPI PUSH4 0x4e487b71 PUSH1 0xe0 SHL PUSH1 0x0 MSTORE PUSH1 0x22 PUSH1 0x4 MSTORE PUSH1 0x24 PUSH1 0x0 REVERT JUMPDEST POP SWAP2 SWAP1 POP JUMP JUMPDEST PUSH4 0x4e487b71 PUSH1 0xe0 SHL PUSH1 0x0 MSTORE PUSH1 0x11 PUSH1 0x4 MSTORE PUSH1 0x24 PUSH1 0x0 REVERT INVALID LOG2 PUSH5 0x6970667358 INVALID SLT SHA3 INVALID TIMESTAMP INVALID DUP6 STATICCALL INVALID REVERT INVALID ADDMOD DUP5 SGT DUP1 INVALID INVALID DELEGATECALL INVALID INVALID INVALID INVALID INVALID INVALID PUSH9 0xfab00163eaf92fa4f4 SWAP11 PUSH5 0x736f6c6343 STOP ADDMOD STOP STOP CALLER']

new_label = truncate_list(label, 2048)
print(new_label)
print(len(new_label))
print(len(new_label[0]))