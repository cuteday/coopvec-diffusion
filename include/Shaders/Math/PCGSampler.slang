#pragma once

struct PCG32
{
    uint64_t state, inc;

    // Initialize from initial state value and stream index
	__init(uint64_t init_state, uint64_t init_seq)
    {
		state = 0;
		inc = (init_seq << 1u) | 1u;
		NextUInt();
		state += init_state;
		NextUInt();
    }

    // Generates random 32 bit unsigned integer
    [mutating]
    uint NextUInt()
    {
		uint64_t oldstate = state;
		state = oldstate * 0x5851f42d4c957f2dULL + inc;
		uint xorshifted = (uint) (((oldstate >> 18u) ^ oldstate) >> 27u);
		uint rot = (uint) (oldstate >> 59u);
		return (xorshifted >> rot) | (xorshifted << ((~rot + 1u) & 31));
    }

	// Generates random float
    [mutating]
	float NextFloat()
    {
		uint x = (NextUInt() >> 9) | 0x3f800000u;
		return reinterpret<float>(x) - 1.0f;
	}
};
