/**
 * Copyright 2013-2023 Software Radio Systems Limited
 *
 * This file is part of srsRAN.
 *
 * srsRAN is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as
 * published by the Free Software Foundation, either version 3 of
 * the License, or (at your option) any later version.
 *
 * srsRAN is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * A copy of the GNU Affero General Public License can be found in
 * the LICENSE file in the top-level directory of this distribution
 * and at http://www.gnu.org/licenses/.
 *
 */

#include "srsenb/hdr/stack/mac/schedulers/sched_time_cust.h"
#include <algorithm>
#include <cstdlib>
#include <limits>

namespace srsenb {

namespace {

/// Number of random draws granted per TTI to the UE with the smallest RNTI, and to every other UE.
const uint32_t PRIORITY_UE_NOF_SLOTS = 100;
const uint32_t DEFAULT_NOF_SLOTS     = 10;
/// A single draw succeeds with probability 1 / SLOT_SUCCESS_DIVISOR.
const int SLOT_SUCCESS_DIVISOR = 100;

/// Returns the smallest RNTI among the UEs configured in the given carrier.
uint16_t find_smallest_rnti(sched_ue_list& ue_db, uint32_t enb_cc_idx)
{
  uint16_t smallest_rnti = std::numeric_limits<uint16_t>::max();
  for (auto& ue_pair : ue_db) {
    sched_ue& user = *ue_pair.second;
    if (user.enb_to_ue_cc_idx(enb_cc_idx) >= 0) {
      smallest_rnti = std::min(smallest_rnti, user.get_rnti());
    }
  }
  return smallest_rnti;
}

/// Decides randomly whether the UE may be scheduled for a new transmission in this TTI.
/// The UE holding the smallest RNTI gets more draws, hence a higher probability of being scheduled.
bool wins_tx_slot(sched_ue& user, uint16_t smallest_rnti)
{
  const uint32_t nof_slots = user.get_rnti() == smallest_rnti ? PRIORITY_UE_NOF_SLOTS : DEFAULT_NOF_SLOTS;
  for (uint32_t slot_number = 0; slot_number < nof_slots; ++slot_number) {
    if (rand() < RAND_MAX / SLOT_SUCCESS_DIVISOR) {
      return true;
    }
  }
  return false;
}

} // namespace

sched_time_cust::sched_time_cust(const sched_cell_params_t&           cell_params_,
                                 const sched_interface::sched_args_t& sched_args)
{
  cc_cfg = &cell_params_;
}

/*****************************************************************
 *                         Downlink
 *****************************************************************/

void sched_time_cust::sched_dl_users(sched_ue_list& ue_db, sf_sched* tti_sched)
{
  if (ue_db.empty()) {
    return;
  }

  // give priority in a time-domain RR basis.
  uint32_t priority_idx = tti_sched->get_tti_tx_dl().to_uint() % (uint32_t)ue_db.size();
  sched_dl_retxs(ue_db, tti_sched, priority_idx);
  sched_dl_newtxs(ue_db, tti_sched, priority_idx);
}

void sched_time_cust::sched_dl_retxs(sched_ue_list& ue_db, sf_sched* tti_sched, size_t prio_idx)
{
  auto iter = ue_db.begin();
  std::advance(iter, prio_idx);
  for (uint32_t ue_count = 0; ue_count < ue_db.size(); ++iter, ++ue_count) {
    if (iter == ue_db.end()) {
      iter = ue_db.begin(); // wrap around
    }
    sched_ue&           user = *iter->second;
    const dl_harq_proc* h    = get_dl_retx_harq(user, tti_sched);
    // Check if there is a pending retx
    if (h == nullptr) {
      continue;
    }
    try_dl_retx_alloc(*tti_sched, user, *h);
  }
}

void sched_time_cust::sched_dl_newtxs(sched_ue_list& ue_db, sf_sched* tti_sched, size_t prio_idx)
{
  const uint16_t smallest_rnti = find_smallest_rnti(ue_db, cc_cfg->enb_cc_idx);

  auto iter = ue_db.begin();
  std::advance(iter, prio_idx);
  for (uint32_t ue_count = 0; ue_count < ue_db.size(); ++iter, ++ue_count) {
    if (iter == ue_db.end()) {
      iter = ue_db.begin(); // wrap around
    }
    sched_ue& user = *iter->second;
    if (user.enb_to_ue_cc_idx(cc_cfg->enb_cc_idx) < 0) {
      continue;
    }
    if (not wins_tx_slot(user, smallest_rnti)) {
      continue;
    }
    const dl_harq_proc* h = get_dl_newtx_harq(user, tti_sched);
    // Without an empty harq for the newtx, no further UE is served in this TTI
    if (h == nullptr) {
      break;
    }
    alloc_result ret = try_dl_newtx_alloc_greedy(*tti_sched, user, *h);
    if (ret == alloc_result::no_cch_space) {
      logger.info("SCHED: Couldn't find space in PDCCH/PUCCH for DL tx for rnti=0x%x", user.get_rnti());
    } else if (ret == alloc_result::success) {
      logger.info("SCHED: allocated DL slot for UE %d", ue_count);
    } else {
      logger.info("SCHED: couldn't allocate DL slot for UE %d because of %s", ue_count, to_string(ret));
    }
  }
}

/*****************************************************************
 *                         Uplink
 *****************************************************************/

void sched_time_cust::sched_ul_users(sched_ue_list& ue_db, sf_sched* tti_sched)
{
  if (ue_db.empty()) {
    return;
  }
  // give priority in a time-domain RR basis.
  uint32_t priority_idx = tti_sched->get_tti_tx_ul().to_uint() % (uint32_t)ue_db.size();
  sched_ul_retxs(ue_db, tti_sched, priority_idx);
  sched_ul_newtxs(ue_db, tti_sched, priority_idx);
}

void sched_time_cust::sched_ul_retxs(sched_ue_list& ue_db, sf_sched* tti_sched, size_t prio_idx)
{
  auto iter = ue_db.begin();
  std::advance(iter, prio_idx);
  for (uint32_t ue_count = 0; ue_count < ue_db.size(); ++iter, ++ue_count) {
    if (iter == ue_db.end()) {
      iter = ue_db.begin(); // wrap around
    }
    sched_ue&           user = *iter->second;
    const ul_harq_proc* h    = get_ul_retx_harq(user, tti_sched);
    // Check if there is a pending retx
    if (h == nullptr) {
      continue;
    }
    alloc_result code = try_ul_retx_alloc(*tti_sched, user, *h);
    if (code == alloc_result::no_cch_space) {
      logger.debug("SCHED: Couldn't find space in PDCCH for UL retx of rnti=0x%x", user.get_rnti());
    }
  }
}

void sched_time_cust::sched_ul_newtxs(sched_ue_list& ue_db, sf_sched* tti_sched, size_t prio_idx)
{
  const uint16_t smallest_rnti = find_smallest_rnti(ue_db, cc_cfg->enb_cc_idx);

  auto iter = ue_db.begin();
  std::advance(iter, prio_idx);
  for (uint32_t ue_count = 0; ue_count < ue_db.size(); ++iter, ++ue_count) {
    if (iter == ue_db.end()) {
      iter = ue_db.begin(); // wrap around
    }
    sched_ue& user = *iter->second;

    // Allocate only if UL carrier is enabled
    bool ul_disabled = false;
    for (auto& i : user.get_ue_cfg().supported_cc_list) {
      if (i.enb_cc_idx == cc_cfg->enb_cc_idx) {
        ul_disabled = i.ul_disabled;
        break;
      }
    }
    if (ul_disabled) {
      continue;
    }
    if (not wins_tx_slot(user, smallest_rnti)) {
      continue;
    }

    // Without an empty harq, pending data or free PRBs for this UE, no further UE is served in this TTI
    const ul_harq_proc* h = get_ul_newtx_harq(user, tti_sched);
    if (h == nullptr) {
      break;
    }
    uint32_t pending_data = user.get_pending_ul_new_data(tti_sched->get_tti_tx_ul(), cc_cfg->enb_cc_idx);
    if (pending_data == 0) {
      break;
    }
    uint32_t     pending_rb = user.get_required_prb_ul(cc_cfg->enb_cc_idx, pending_data);
    prb_interval alloc      = find_contiguous_ul_prbs(pending_rb, tti_sched->get_ul_mask());
    if (alloc.empty()) {
      break;
    }

    alloc_result ret = tti_sched->alloc_ul_user(&user, alloc);
    if (ret == alloc_result::no_cch_space) {
      logger.info(
          "SCHED: rnti=0x%x, cc=%d, Couldn't find space in PDCCH for UL tx", user.get_rnti(), cc_cfg->enb_cc_idx);
    } else if (ret == alloc_result::success) {
      logger.info("SCHED: allocated UL slot for UE %d", ue_count);
    } else {
      logger.info("SCHED: couldn't allocate UL slot for UE %d because of %s", ue_count, to_string(ret));
    }
  }
}

} // namespace srsenb