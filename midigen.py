def floorN(num, divisor):
    return num - (num%divisor)

NOTE_MAX = 108
NOTE_MIN = 21
NB_NOTES = NOTE_MAX - NOTE_MIN + 1

TIME_INC = 8
TIME_MAX = 1000
NB_TIME = TIME_MAX // TIME_INC

VEL_MIN = 0
VEL_MAX = 124
VEL_INC = 4
NB_VEL = (VEL_MAX - VEL_MIN) // VEL_INC + 1

ON_INDEX = 0
OFF_INDEX = ON_INDEX + NB_NOTES
TIME_INDEX = OFF_INDEX + NB_NOTES
VEL_INDEX = TIME_INDEX + NB_TIME

def generate_from_seq(seq, name):
    f = open(name, mode='w')

    f.write("0, 0, Header, 1, 2, 384\n")
    f.write("1, 0, Start_track\n")
    f.write("1, 0, Tempo, 500000\n")
    f.write("1, 0, Time_signature, 4, 2, 24, 8\n")
    f.write("1, 1, End_track\n")
    f.write("2, 0, Start_track\n")
    f.write("2, 0, Program_c, 0, 0\n")

    c_time = 0
    c_vel = 0

    for s in seq:
        if ON_INDEX <= s and s < OFF_INDEX: # Note on event
            note = NOTE_MIN + (s - ON_INDEX)
            f.write(f"2, {c_time}, Note_on_c, 0, {note}, {c_vel}\n")
        elif OFF_INDEX <= s and s < TIME_INDEX: # Note off event
            note = NOTE_MIN + (s - OFF_INDEX)
            f.write(f"2, {c_time}, Note_on_c, 0, {note}, 0\n")
        elif TIME_INDEX <= s and s < VEL_INDEX: # Time skip event
            d_t = (s - TIME_INDEX + 1) * TIME_INC
            c_time += d_t
        else: # Velocity change event
            n_v = (s - VEL_INDEX) * VEL_INC
            c_vel = n_v

    f.write(f"2, {c_time+1}, End_track\n")
    f.write("0, 0, End_of_file")
    
    f.close()
