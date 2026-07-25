#include "duckdb/planner/binder.hpp"
#include "duckdb/planner/bound_result_modifier.hpp"
#include "duckdb/planner/expression/bound_cast_expression.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/planner/operator/logical_set_operation.hpp"
#include "duckdb/planner/query_node/bound_set_operation_node.hpp"
#include "duckdb/function/table/read_csv.hpp"
#include "duckdb/planner/operator/logical_get.hpp"

namespace duckdb {

// Optionally push a PROJECTION operator
unique_ptr<LogicalOperator> Binder::CastLogicalOperatorToTypes(const vector<LogicalType> &source_types,
                                                               const vector<LogicalType> &target_types,
                                                               unique_ptr<LogicalOperator> op) {
	D_ASSERT(op);
	D_ASSERT(source_types.size() == target_types.size());
	auto node = op.get();
	if (source_types == target_types) {
		bool has_cast = false;
		if (node->type == LogicalOperatorType::LOGICAL_PROJECTION) {
			for (auto &expression : node->expressions) {
				if (BoundCastExpression::IsCast(*expression)) {
					has_cast = true;
					break;
				}
			}
		}
		if (!has_cast) {
			return op;
		}
	}
	if (node->type == LogicalOperatorType::LOGICAL_PROJECTION) {
		D_ASSERT(node->expressions.size() == source_types.size());
		// add the casts to the selection list
		for (idx_t i = 0; i < target_types.size(); i++) {
			if (source_types[i] != target_types[i]) {
				// differing types, have to add a cast
				auto cur_alias = node->expressions[i]->GetAlias();
				node->expressions[i] =
				    BoundCastExpression::AddCastToType(context, std::move(node->expressions[i]), target_types[i]);
				node->expressions[i]->SetAlias(cur_alias);
			}
		}
		return op;
	} else {
		if (source_types == target_types) {
			return op;
		}
		// found a non-projection operator
		// push a new projection containing the casts

		// fetch the set of column bindings
		auto setop_columns = op->GetColumnBindings();
		D_ASSERT(setop_columns.size() == source_types.size());

		// now generate the expression list
		vector<unique_ptr<Expression>> select_list;
		for (idx_t i = 0; i < target_types.size(); i++) {
			unique_ptr<Expression> result = make_uniq<BoundColumnRefExpression>(source_types[i], setop_columns[i]);
			if (source_types[i] != target_types[i]) {
				// add a cast only if the source and target types are not equivalent
				result = BoundCastExpression::AddCastToType(context, std::move(result), target_types[i]);
			}
			select_list.push_back(std::move(result));
		}
		auto projection = make_uniq<LogicalProjection>(GenerateTableIndex(), std::move(select_list));
		projection->children.push_back(std::move(op));
		return std::move(projection);
	}
}

//! Set operations with set semantics are bound with a DISTINCT modifier on top of the set operation
//! Check if that modifier deduplicates over all columns of the set operation
static bool DistinctHandledByModifier(BoundSetOperationNode &node) {
	for (auto &modifier : node.modifiers) {
		if (modifier->type != ResultModifierType::DISTINCT_MODIFIER) {
			continue;
		}
		auto &distinct = modifier->Cast<BoundDistinctModifier>();
		if (distinct.distinct_type != DistinctType::DISTINCT) {
			continue;
		}
		// the modifier must deduplicate over all columns of the set operation
		if (distinct.target_distincts.size() == node.types.size()) {
			return true;
		}
	}
	return false;
}

unique_ptr<LogicalOperator> Binder::CreatePlan(BoundSetOperationNode &node) {
	// create actual logical ops for setops
	LogicalOperatorType logical_type = LogicalOperatorType::LOGICAL_INVALID;
	bool setop_all = node.setop_all;
	switch (node.setop_type) {
	case SetOperationType::UNION:
	case SetOperationType::UNION_BY_NAME:
		logical_type = LogicalOperatorType::LOGICAL_UNION;
		// a UNION with a DISTINCT modifier on top is equivalent to a UNION ALL followed by that modifier
		// use bag semantics for the union itself so that we don't deduplicate twice
		if (!setop_all && DistinctHandledByModifier(node)) {
			setop_all = true;
		}
		break;
	case SetOperationType::EXCEPT:
		logical_type = LogicalOperatorType::LOGICAL_EXCEPT;
		break;
	case SetOperationType::INTERSECT:
		logical_type = LogicalOperatorType::LOGICAL_INTERSECT;
		break;
	default:
		throw InternalException("Unsupported logical operator type for set-operation");
	}
	// Generate the logical plan for the children of the set operation

	D_ASSERT(node.bound_children.size() >= 2);
	vector<unique_ptr<LogicalOperator>> children;
	for (idx_t child_idx = 0; child_idx < node.bound_children.size(); child_idx++) {
		auto &child = node.bound_children[child_idx];
		auto &child_binder = *node.child_binders[child_idx];

		// construct the logical plan for the child node
		auto child_node = std::move(child.plan);
		// push casts for the target types
		child_node = CastLogicalOperatorToTypes(child.types, node.types, std::move(child_node));
		// check if there are any unplanned subqueries left in any child
		if (child_binder.has_unplanned_dependent_joins) {
			has_unplanned_dependent_joins = true;
		}
		children.push_back(std::move(child_node));
	}
	auto root = make_uniq<LogicalSetOperation>(node.setop_index, node.types.size(), std::move(children), logical_type,
	                                           setop_all);
	return VisitQueryNode(node, std::move(root));
}

} // namespace duckdb
