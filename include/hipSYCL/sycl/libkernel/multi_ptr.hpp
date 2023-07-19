/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2018 Aksel Alpay
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef HIPSYCL_MULTI_PTR_HPP
#define HIPSYCL_MULTI_PTR_HPP

#include "hipSYCL/sycl/libkernel/backend.hpp"
#include "hipSYCL/sycl/access.hpp"
#include "hipSYCL/sycl/libkernel/memory.hpp"

#include <type_traits>

namespace hipsycl {
namespace sycl {

template<typename dataT, int dimensions,
         access::mode accessmode,
         access::target accessTarget,
         access::placeholder isPlaceholder>
class accessor;

template <typename T> 
struct remove_decoration 
{
  using type = T;
};

template <typename T> 
using remove_decoration_t = remove_decoration<T>::type;

namespace detail {

  // TODO hipSYCL does not differentiate between the address spaces.
  template <typename ElementType, access::address_space Space>
  struct DecoratedType 
  {
    using type = ElementType;
  };

 template <typename ToT, typename FromT> 
 inline ToT cast(FromT from)
 {
   return reinterpret_cast<ToT>(from); 
 }

} // namespace detail

template <typename ElementType, access::address_space Space, access::decorated DecorateAddress>
class multi_ptr
{
public:
  
  //TODO should the type be made private? It doesn't look any different to value_type below.
  using decorated_type = typename detail::DecoratedType<ElementType, Space>::type;

  static constexpr bool is_decorated = 
    DecorateAddress == access::decorated::yes;
  static constexpr access::address_space address_space = Space;

  using value_type = ElementType;
  using pointer = std::conditional_t<is_decorated, decorated_type*,
                                     std::add_pointer_t<value_type>>;
  using reference = std::conditional_t<is_decorated,decorated_type&,
                                       std::add_lvalue_reference_t<value_type>>;
  using iterator_category = std::random_access_iterator_tag;
  using difference_type == std::ptrdiff_t;

  static_assert(std::is_same_v<remove_decoration_t<pointer>,
                               std::add_pointer_t<value_type>>);
  static_assert(std::is_same_v<remove_decoration_t<reference>,
                               std::add_lvalue_reference_t<value_type>>);

  // Legacy has a different interface.
  static_assert(DecorateAddress != access::decorated::legacy);

  // Constructors
  HIPSYCL_UNIVERSAL_TARGET
  multi_ptr()
    : _ptr{nullptr}
  {}

  multi_ptr(const multi_ptr&) = default;
  multi_ptr(multi_ptr&&) = default;

  HIPSYCL_UNIVERSAL_TARGET explicit multi_ptr(
    typename multi_ptr<ElementType, Space, access::decorated::yes>::pointer ptr)
    : _ptr{ptr}
  {}

  HIPSYCL_UNIVERSAL_TARGET
  multi_ptr(std::nullptr_t)
    : _ptr{nullptr}
  {}

  template <int Dimensions, access_mode Mode, access::placeholder IsPlaceholder,
       typename = typename std::enable_if_t<Space == access::address_space::global_space ||
                                   Space == access::address_space::generic_space>>
  HIPSYCL_UNIVERSAL_TARGET
  multi_ptr(
    accessor<value_type, Dimensions, Mode, target::device, IsPlaceholder> a)
    : _ptr{a.get_pointer()}
  {}

  template <int Dimensions,  
       typename = typename std::enable_if_t<Space == access::address_space::local_space ||
                                   Space == access::address_space::generic_space>>
  HIPSYCL_UNIVERSAL_TARGET
  multi_ptr(local_accessor<ElementType, Dimensions> a)
    : _ptr{a.get_pointer()}
  {}

  [[deprecated("Deprecated since SYCL 2020, use the overload with "
               "local_accessor instead.")]]
  template <int Dimensions, access_mode Mode, access::placeholder IsPlaceholder, 
       typename = typename std::enable_if_t<Space == access::address_space::local_space ||
                                   Space == access::address_space::generic_space>>
  HIPSYCL_UNIVERSAL_TARGET
  multi_ptr(
    accessor<value_type, Dimensions, Mode, target::local, IsPlaceholder> a)
    : _ptr{a.get_pointer()}
  {}

  // TODO:DONE? Add impl for deprecated constructor make_ptr
  // TODO:DONE? Add impl for constructor address_space_cast

  // Assignment and access operators
  HIPSYCL_UNIVERSAL_TARGET
  multi_ptr& operator=(const multi_ptr& other)
  { 
    _ptr = other._ptr;
    return *this;
  }

  HIPSYCL_UNIVERSAL_TARGET
  multi_ptr& operator=(multi_ptr&& other)
  { 
    _ptr = other._ptr;
    return *this;
  }

  HIPSYCL_UNIVERSAL_TARGET
  multi_ptr& operator=(std::nullptr_t)
  { 
    _ptr = nullptr;
    return *this;
  }

  template <access::address_space AS, access::decorated IsDecorated,
      typename  = typename std::enable_if_t<Space == access::address_space::generic_space && 
                                   AS != access::addresss_space::constant_space>>
  HIPSYCL_UNIVERSAL_TARGET
  multi_ptr& operator=(const multi_ptr<value_type, AS, IsDecorated>& other)
  {
    _ptr = detail::cast<decorated_type *>(other.get_decorated());
    return *this;
  }

  template <access::address_space AS, access::decorated IsDecorated
      typename  = typename std::enable_if_t<Space == access::address_space::generic_space && 
                                   AS != access::addresss_space::constant_space>>
  HIPSYCL_UNIVERSAL_TARGET
  multi_ptr& operator=(multi_ptr<value_type, AS, IsDecorated>&& other)
  {
    _ptr = detail::cast<decorated_type *>(std::move(other._ptr));
    return *this;
  }


  HIPSYCL_UNIVERSAL_TARGET
  reference operator[](difference_type index) const
  {
    return *_ptr[index];
  }


  HIPSYCL_UNIVERSAL_TARGET
  reference operator*() const
  {
    return *_ptr;
  }

  //TODO Optional template spec, since class is assumes value_type is not void!
  template <typename = typename std::enable_if_t<!std::is_void_v<value_type>>>
  HIPSYCL_UNIVERSAL_TARGET
  pointer operator->() const
  {
    return get();
  }

  HIPSYCL_UNIVERSAL_TARGET
  pointer get() const
  { 
    return detail::cast<pointer>(_ptr);
  }

  std::add_pointer_t<value_type> get_raw() const
  {
    return reinterpret_cast<std::add_pointer_t<value_type>>(get());
  }

  decorated_type* get_decorated() const
  {
    return _ptr;
  }
  
  [[deprecated ("Conversion to underlying pointer type is deprecated since SYCL 2020."
                " Use get() instead.")]]
  HIPSYCL_UNIVERSAL_TARGET
  operator pointer() const
  {
    return get();
  }

  // Cast to private_ptr
  template <access::address_space AS, access::decorated IsDecorated,
      typename  = typename std::enable_if_t<Space == access::address_space::generic_space>>
  HIPSYCL_UNIVERSAL_TARGET
  explicit operator multi_ptr<value_type, access::address_space::private_space,
                              IsDecorated>() const
  {
    return multi_ptr<value_type, AS, IsDecorated>
    {
      detail::cast<typename multi_ptr<value_type, AS, 
                   access::decorated::yes>::pointer>(get_decorated())
    };
  }

  // Cast to private_ptr
  template <access::address_space AS, access::decorated IsDecorated,
      typename  = typename std::enable_if_t<Space == access::address_space::generic_space>>
  HIPSYCL_UNIVERSAL_TARGET
  explicit operator multi_ptr<const value_type, access::address_space::private_space,
                              IsDecorated>() const
  {
    return multi_ptr<const value_type, AS, IsDecorated>
    {
      detail::cast<typename multi_ptr<const value_type, AS,
                   access::decorated::yes>::pointer>(get_decorated())
    };
  }

  // Cast to global_ptr
  template <access::address_space AS, access::decorated IsDecorated,
      typename  = typename std::enable_if_t<Space == access::address_space::generic_space>>
  HIPSYCL_UNIVERSAL_TARGET
  explicit operator multi_ptr<value_type, access::address_space::global_space,
                              IsDecorated>() const
  {
    return multi_ptr<value_type, AS, IsDecorated>
    {
      detail::cast<typename multi_ptr<value_type, AS, 
                   access::decorated::yes>::pointer>(get_decorated()) 
    };
  }

  // Cast to global_ptr
  template <access::address_space AS, access::decorated IsDecorated,
      typename  = typename std::enable_if_t<Space == access::address_space::generic_space>>
  HIPSYCL_UNIVERSAL_TARGET
  explicit operator multi_ptr<const value_type, access::address_space::global_space,
                              IsDecorated>() const
  {
    return multi_ptr<const value_type, AS, IsDecorated>
    {
      detail::cast<typename multi_ptr<const value_type, AS,
                   access::decorated::yes>::pointer>(get_decorated())
    };
  }

  // Cast to local_ptr 
  template <access::address_space AS, access::decorated IsDecorated,
      typename  = typename std::enable_if_t<Space == access::address_space::generic_space>>
  HIPSYCL_UNIVERSAL_TARGET
  explicit operator multi_ptr<value_type, access::address_space::local_space,
                              IsDecorated>() const
  {
    return multi_ptr<value_type, AS, IsDecorated>
    {
      detail::cast<typename multi_ptr<value_type, AS,
                   access::decorated::yes>::pointer>(get_decorated())
    };
  }

  // Cast to local_ptr
  template <access::address_space AS, access::decorated IsDecorated,
      typename  = typename std::enable_if_t<Space == access::address_space::generic_space>>
  HIPSYCL_UNIVERSAL_TARGET
  explicit operator multi_ptr<const value_type, access::address_space::local_space,
                              IsDecorated>() const
  {
    return mutli_ptr<const value_type, AS, IsDecorated>
    {
      detail::cast<typename multi_ptr<const value_type, AS,
                   access::decorated::yes>::pointer>(get_decorated())
    };
  }

  // Implicit conversion to a multi_ptr<void>.
  // Available only when: (!std::is_const_v<value_type>)
  template <access::decorated IsDecorated,
	  typename = typename std::enable_if_t<Space !=std::is_const_v<value_type>>>
  HIPSYCL_UNIVERSAL_TARGET
  operator multi_ptr<void, Space, IsDecorated>() const
  {
    return multi_ptr<void, Space, IsDecorated>
    {
      detail::cast<typename mutli_ptr<void, Space,
                   access::decorated::yes>::pointer>(get_decorated())
    };
  }

  // Implicit conversion to a multi_ptr<void>.
  // Available only when: (std::is_const_v<value_type>)
  template <access::decorated IsDecorated,
	   typename = typename std::enable_if_t<Space == std::is_const_v<value_type>>>
  HIPSYCL_UNIVERSAL_TARGET
  operator multi_ptr<const void, Space, IsDecorated>() const
  {
    return muulti_ptr<const void, Space, IsDecorated>
      {
        detail::cast<typename multi_ptr<const void, Space,
	             access::decorated::yes>::pointer>(get_decorated())
      };
  }

  // Implicit conversion to multi_ptr<const value_type, Space>.
  template <access::decorated IsDecorated>
  HIPSYCL_UNIVERSAL_TARGET
  operator multi_ptr<const value_type, Space, IsDecorated>() const
  {
    return multi_ptr<const value_type, Space, IsDecorated>
      {
        detail::cast<typename multi_ptr<const value_type, Space,
	             access::decorated::yes>::pointer>(get_decorated())
      };
  }

  // Implicit conversion to the non-decorated version of multi_ptr.
  // Available only when: (is_decorated == true)
  
  // TODO is_decorated is the boolean associated with DecorateAddress.
  // Should the IsDecorated be replaced by DecorateAddress?
  template <access::decorated IsDecorated,
	   typename = typename std::enable_if_t<is_decorated>>
  HIPSYCL_UNIVERSAL_TARGET
  operator multi_ptr<value_type, Space, access::decorated::no>() const
  {
    return multi_ptr<value_type, Space, access::decorated::no>
    {
      get_decorated()
    };
  }

  // Implicit conversion to the decorated version of multi_ptr.
  // Available only when: (is_decorated == false)
  // TODO is_decorated is the boolean associated with DecorateAddress.
  // Should the IsDecorated be replaced by DecorateAddress?
  template <access::decorated IsDecorated,
	   typename = typename std::enable_if_t<!is_decorated>>
  HIPSYCL_UNIVERSAL_TARGET
  operator multi_ptr<value_type, Space, access::decorated::yes>() const
  {
    return multi_ptr<value_type, Space, access::decorated::yes>
    {
      get_decorated()
    };
  }

  // Available only when: (Space == address_space::global_space)
  template < access::address_space Space,
	   typename = typename std::enable_if_t<
		       Space == access::address_space::global_space>>
  HIPSYCL_UNIVERSAL_TARGET
  void prefetch(size_t numElements) const
  {
    size_t sizeElements = numElements * sizeof(value_type);
    using ptr_t = 
	    typename detail::DecoratedType<char, Space>::type const *;
    reinterpret_cast<ptr_t>(get_decorated(), sizeElements);
  }

  // Arithmatic operators
  HIPSYCL_UNIVERSAL_TARGET
  friend multi_ptr& operator++(multi_ptr& mp)
  {
    ++(mp._ptr);
    return mp;
  }

  HIPSYCL_UNIVERSAL_TARGET
  friend multi_ptr operator++(multi_ptr& mp, int)
  {
    multi_ptr old = mp;
    ++(mp._ptr);
    return old;
  }

  HIPSYCL_UNIVERSAL_TARGET
  friend multi_ptr& operator--(multi_ptr& mp)
  {
    --(mp.ptr);
    return mp; 
  }

  HIPSYCL_UNIVERSAL_TARGET
  friend multi_ptr operator--(multi_ptr& mp, int)
  {
    multi_ptr old = mp;
    --(mp._ptr);
    return old;
  }

  HIPSYCL_UNIVERSAL_TARGET
  friend multi_ptr& operator+=(multi_ptr& lhs, difference_type r)
  {
    lhs._ptr += r;
    return lhs;
  }

  HIPSYCL_UNIVERSAL_TARGET
  friend multi_ptr& operator-=(multi_ptr& lhs, difference_type r)
  {
    lhs._ptr -= r;
    return lhs;
  }

  HIPSYCL_UNIVERSAL_TARGET
  friend multi_ptr operator+(const multi_ptr& lhs, difference_type r)
  {
    return multi_ptr{lhs._ptr + r};
  }  

  HIPSYCL_UNIVERSAL_TARGET
  friend multi_ptr operator-(const multi_ptr& lhs, difference_type r)
  {
    return multi_ptr{lhs._ptr - r};
  }

  HIPSYCL_UNIVERSAL_TARGET
  friend reference operator*(const multi_ptr& lhs)
  {
  }


  HIPSYCL_UNIVERSAL_TARGET
  friend bool operator==(const multi_ptr& lhs, const multi_ptr& rhs)
  {
  }

  HIPSYCL_UNIVERSAL_TARGET
  friend bool operator!=(const multi_ptr& lhs, const multi_ptr& rhs)
  {
  }

  friend bool operator<(const multi_ptr& lhs, const multi_ptr& rhs)
  {
  }

  friend bool operator>(const multi_ptr& lhs, const multi_ptr& rhs)
  {
  }

  friend bool operator<=(const multi_ptr& lhs, const multi_ptr& rhs)
  {
  }

  friend bool operator>=(const multi_ptr& lhs, const multi_ptr& rhs)
  {
  }


  friend bool operator==(const multi_ptr& lhs, std::nullptr_t)
  {
  }

  friend bool operator!=(const multi_ptr& lhs, std::nullptr_t)
  {
  }

  friend bool operator<(const multi_ptr& lhs, std::nullptr_t)
  {
  }

  friend bool operator>(const multi_ptr& lhs, std::nullptr_t)
  {
  }

  friend bool operator<=(const multi_ptr& lhs, std::nullptr_t)
  {
  }

  friend bool operator>=(const multi_ptr& lhs, std::nullptr_t)
  {
  }


  friend bool operator==(std::nullptr_t, const multi_ptr& rhs)
  {
  }

  friend bool operator!=(std::nullptr_t, const multi_ptr& rhs)
  {
  }

  friend bool operator<(std::nullptr_t, const multi_ptr& rhs)
  {
  }

  friend bool operator>(std::nullptr_t, const multi_ptr& rhs)
  {
  }

  friend bool operator<=(std::nullptr_t, const multi_ptr& rhs)
  {
  }

  friend bool operator>=(std::nullptr_t, const multi_ptr& rhs)
  {
  }

private:
  decorated_type * _ptr;

};


// Specialization of multi_ptr for void and const void
// VoidType can be either void or const void
template <access::address_space Space, access::decorated DecorateAddress>
class multi_ptr<VoidType, Space, DecorateAddress>
{
  // TODO
};


// Deprecated, address_space_cast should be used instead.
template <typename ElementType, access::address_space Space,
           access::decorated DecorateAddress>
multi_ptr<ElementType, Space, DecorateAddress> make_ptr(ElementType*);

template <access::address_space Space, access::decorated DecorateAddress,
          typename ElementType>
multi_ptr<ElementType, Space, DecorateAddress> address_space_cast (ElementType*);


// Deduction guides
template <typename T, int Dimensions, access_mode Mode, access::placeholder IsPlaceholder>
multi_ptr(accessor<T, Dimensions, Mode, target::device, IsPlaceholder>)
         -> multi_ptr<T, access::address_space::global_space, access::decorated::no>;

template <typename T, int Dimensions>
multi_ptr(local_accessor<T, Dimensions>)
          -> multi_ptr<T, access::address_space::local_space, access::decorated::no>;

} // namespace sycl
} // namespace hipsycl

#endif
